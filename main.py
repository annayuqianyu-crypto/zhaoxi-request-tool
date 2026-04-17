import json
import re
import io
import os
import secrets
from datetime import datetime
from typing import List, Union, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from fastapi import FastAPI, HTTPException, UploadFile, File, Response, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = OpenAI(
    api_key=os.environ.get("WOLFAI_API_KEY"),
    base_url="https://wolfai.top/v1"
)

# ─────────────────────────────────────────────
# 数据库初始化（PostgreSQL / Supabase）
# ─────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "")

def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id          TEXT PRIMARY KEY,
        email       TEXT UNIQUE NOT NULL,
        name        TEXT NOT NULL,
        token       TEXT UNIQUE NOT NULL,
        is_admin    INTEGER DEFAULT 0,
        created_at  TEXT NOT NULL,
        last_active TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id           TEXT PRIMARY KEY,
        user_id      TEXT NOT NULL,
        title        TEXT,
        history      TEXT,
        requirements TEXT,
        mermaid      TEXT,
        completeness INTEGER DEFAULT 0,
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    )
    """)
    conn.commit()
    cur.close()
    conn.close()

init_db()

def require_token(authorization: Optional[str] = Header(None)):
    """从 Authorization: Bearer <token> 中验证用户"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "未登录，请先登录")
    token = authorization[7:]
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE token=%s", (token,))
    user = cur.fetchone()
    cur.close(); conn.close()
    if not user:
        raise HTTPException(401, "登录已失效，请重新登录")
    return dict(user)

def require_admin(authorization: Optional[str] = Header(None)):
    user = require_token(authorization)
    if not user["is_admin"]:
        raise HTTPException(403, "无管理员权限")
    return user

# ─────────────────────────────────────────────
# System prompt：Grill-Me 风格 + 结构化需求输出
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """你是朝曦金融机构高端客户增值服务平台的资深业务分析师，精通公司五大核心服务体系：
1. 家族财富架构服务（境内外信托架构、资本市场投融资、全球资产配置、股权激励）
2. 全球税务规划（美/加/新加坡税务合规、上市公司股东减持税务、跨境架构涉税、CRS/FATCA）
3. 全球法律咨询（资本金融、公司商事、婚姻家事、企业合规、不良资产处置）
4. 资本市场服务（上市公司股东减持、股东融资退出、拟上市公司架构、私募基金架构）
5. 企业治理（ESG合规、组织效能提升、股权激励体系、接班人规划）

五大客群：上市公司股东、新兴/传统企业家、国际化人士（美/加/澳/英）、企业家太太、高净值家族。

任务：通过深度访谈，将朝曦业务团队模糊的想法转化为清晰、可落地的IT系统需求。

## 访谈方法论

**第一步：场景剧本破冰（方法7）**
不要直接问"你要什么功能"，请业务方讲一个真实工作故事：
"请描述最近您或同事实际遇到的一个具体情境，就像在给我讲故事一样。"

**第二步：结构化深挖（方法1 — 每次只问一个最关键的问题）**
根据故事，判断哪个维度最不清晰，逐一追问：
- 【角色与权限】谁在用？什么职级？权限边界？
- 【触发条件】什么情况下启动流程？频率？
- 【输入数据】需要哪些信息？从哪里来？格式？
- 【处理流程】每个步骤是什么？谁来操作？
- 【审批链路】谁审批？几级？时限要求？
- 【输出结果】产出是什么？发给谁？存在哪里？
- 【异常处理】出错怎么办？有哪些特殊情况？
- 【合规要求】有哪些监管规定或内部制度必须遵守？

**领域专项追问（自动识别业务领域）：**
- 税务场景：申报期限、税种、纳税主体、税务机关对接、凭证留存年限、汇算清缴
- 法律场景：合同类型、审阅层级、印章管控、律师介入时机、诉讼风险分级、文件密级
- 资本市场：交易品种、风控限额、监管报送（证监会/交易所/AMAC）、估值方式、结算周期

**第三步：双轨实时画板（方法11）**
每次回复同步更新流程图，让业务方"看着图纠错"而非凭空想象。

## 流程图生成规范
每次必须输出完整 Mermaid 代码（graph TD），随对话逐步丰富，不要清空已有内容：
```
graph TD
    classDef startEnd fill:#1a365d,stroke:#1a365d,color:#fff,rx:20
    classDef process fill:#ebf4ff,stroke:#2b6cb0,color:#1a365d
    classDef decision fill:#fffaf0,stroke:#ed8936,color:#7b341e
    classDef output fill:#f0fff4,stroke:#38a169,color:#276749
    classDef warning fill:#fff5f5,stroke:#fc8181,color:#742a2a

    A([开始]):::startEnd --> B[流程步骤]:::process
    B --> C{判断条件}:::decision
    C -->|是| D[处理结果]:::output
    C -->|否| E[异常处理]:::warning
```
要求：
- 中文标签，每个标签不超过12字
- 使用 subgraph 对阶段分组（如：申请阶段、审批阶段、归档阶段）
- 决策节点用菱形 {}，开始/结束用圆角 ([])

## 需求清单生成规范
当某功能已经足够清晰时，将其加入 requirements 数组（结构化对象）：
{
  "module": "模块名（如：申请管理）",
  "feature": "功能名（如：采购申请提交）",
  "description": "功能详细描述，说明该功能做什么、解决什么问题",
  "wireframe": "页面布局描述，格式：顶部：...\\n中部：...\\n底部：...",
  "process": ["步骤1：操作者做什么", "步骤2：系统做什么"],
  "inputs": ["输入数据项1", "输入数据项2"],
  "outputs": ["输出结果1", "输出结果2"]
}

## 追问原则
1. 每次只问一个问题，绝不连问
2. 不接受模糊回答：听到"大概""差不多"时追问具体数字或细节
3. requirements 只记录已确认清楚的需求，模糊的继续追问
4. completeness 达 80 以上才建议进入总结
5. 全程使用中文

## 严格输出格式（只输出 JSON，不输出任何其他文字）
{
  "message": "AI回应文字 + 下一个追问问题",
  "mermaid": "完整Mermaid代码字符串",
  "requirements": [ ...结构化需求对象数组... ],
  "stage": "opening 或 exploring 或 drilling 或 summarizing",
  "completeness": 0到100整数
}"""

OPENING_MESSAGE = {
    "message": "您好！我是朝曦的业务需求分析助手，熟悉家族财富架构、全球税务规划、资本市场服务、法律咨询和企业治理五大业务体系。\n\n**我的工作方式：** 通过深度访谈把模糊想法变成清晰的IT需求，右侧实时生成业务流程图，需求清单分四个板块展示：需求梳理、线框图、流程图、PRD文档，最终可直接交付IT团队。\n\n**请按以下框架描述您的场景**（能说多少说多少，其余我来追问）：\n\n🏢 **所在部门**：哪个业务条线或岗位会使用这个系统？\n　　（如：税务团队 / 架构师团队 / 资本市场组 / 客服中台）\n😣 **业务痛点**：目前这件事最大的困难或效率瓶颈是什么？\n🎯 **期望解决**：您希望系统能帮您做到什么，达到什么效果？\n📥 **涉及输入**：需要录入或上传哪些信息？数据从哪里来？\n📤 **期望输出**：系统最终要产出什么？发给谁？存在哪里？\n\n您也可以直接上传相关文件（Word/Excel/PPT）或使用 🎤 语音描述，我会帮您提炼需求。",
    "mermaid": "graph TD\n    classDef startEnd fill:#1a365d,stroke:#1a365d,color:#fff\n    classDef process fill:#ebf4ff,stroke:#2b6cb0,color:#1a365d\n    classDef output fill:#f0fff4,stroke:#38a169,color:#276749\n\n    A([🚀 开始访谈]):::startEnd --> B[描述真实业务场景]:::process\n    B --> C[AI深度追问澄清]:::process\n    C --> D[实时生成流程图]:::process\n    D --> E[积累需求清单]:::process\n    E --> F([导出需求文档]):::startEnd",
    "requirements": [],
    "stage": "opening",
    "completeness": 0
}


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class LoginRequest(BaseModel):
    name: str
    email: str

class SaveSessionRequest(BaseModel):
    session_id: Optional[str] = None
    title: str
    history: list
    requirements: list
    mermaid: str = ""
    completeness: int = 0

class ChatRequest(BaseModel):
    history: List[Message]
    message: str


class ExportRequest(BaseModel):
    history: List[Message]
    requirements: List[Union[dict, str]]
    mermaid: str


class WordExportReq(BaseModel):
    type: str  # "requirements" or "prd"
    requirements: List[Union[dict, str]] = []
    prd_text: str = ""


class GenerateRequest(BaseModel):
    history: List[Message]
    requirements: List[Union[dict, str]]
    type: str  # summary | wireframe | preview | flowchart

class DemoRequest(BaseModel):
    history: List[Message]
    requirements: List[Union[dict, str]] = []
    mermaid: str = ""


# ─────────────────────────────────────────────
# File extraction helpers
# ─────────────────────────────────────────────
def extract_docx(content: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(content))
    texts = []
    for para in doc.paragraphs:
        if para.text.strip():
            texts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
            if row_text:
                texts.append(row_text)
    return "\n".join(texts)


def extract_xlsx(content: bytes) -> str:
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(content), data_only=True)
    texts = []
    for sheet in wb.worksheets:
        texts.append(f"[工作表：{sheet.title}]")
        for row in sheet.iter_rows(values_only=True):
            vals = [str(v) for v in row if v is not None and str(v).strip()]
            if vals:
                texts.append(" | ".join(vals))
    return "\n".join(texts[:300])


def extract_pptx(content: bytes) -> str:
    from pptx import Presentation
    prs = Presentation(io.BytesIO(content))
    texts = []
    for i, slide in enumerate(prs.slides, 1):
        texts.append(f"\n[幻灯片 {i}]")
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    if para.text.strip():
                        texts.append(para.text)
    return "\n".join(texts)


def parse_ai_response(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    # 第一次尝试直接解析
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # 第二次：把 mermaid 字段里的真实换行替换为 \n，再解析
    try:
        fixed = re.sub(
            r'("mermaid"\s*:\s*")(.*?)(")',
            lambda m: m.group(1) + m.group(2).replace('\n', '\\n').replace('\r', '') + m.group(3),
            cleaned, flags=re.DOTALL
        )
        return json.loads(fixed)
    except Exception:
        pass
    # 第三次：正则提取最外层 JSON 对象
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    # 最终 fallback：只提取 message 文字（不泄漏 Mermaid 代码）
    msg_match = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
    fallback_msg = msg_match.group(1).encode().decode('unicode_escape') if msg_match else "AI 正在分析中，请稍候…"
    return {
        "message": fallback_msg,
        "mermaid": "graph TD\n    A[业务场景] --> B[待完善]",
        "requirements": [],
        "stage": "exploring",
        "completeness": 10,
    }


# ─────────────────────────────────────────────
# API routes
# ─────────────────────────────────────────────
@app.get("/api/init")
async def init():
    return OPENING_MESSAGE


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or ""
    content = await file.read()
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    try:
        if ext == "docx":
            text = extract_docx(content)
        elif ext == "xlsx":
            text = extract_xlsx(content)
        elif ext in ("pptx", "ppt"):
            text = extract_pptx(content)
        else:
            raise HTTPException(400, "仅支持 .docx、.xlsx、.pptx 格式")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"文件解析失败：{e}")

    MAX = 8000
    if len(text) > MAX:
        text = text[:MAX] + f"\n…（内容过长，已截取前 {MAX} 字）"

    return {"filename": filename, "extracted_text": text, "char_count": len(text)}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.history]
    messages.append({"role": "user", "content": req.message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2048,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        )
    except Exception as e:
        raise HTTPException(500, f"AI API 错误：{e}")

    return parse_ai_response(response.choices[0].message.content)


@app.post("/api/export")
async def export_doc(req: ExportRequest):
    history_text = "\n\n".join(
        f"**{'业务方' if m.role == 'user' else '分析师'}**：{m.content}"
        for m in req.history
    )

    req_lines = []
    for r in req.requirements:
        if isinstance(r, dict):
            req_lines.append(f"\n### {r.get('module','')} · {r.get('feature','')}")
            req_lines.append(f"**功能说明**：{r.get('description','')}")
            if r.get("wireframe"):
                req_lines.append(f"**界面草图**：{r['wireframe']}")
            if r.get("process"):
                req_lines.append("**处理流程**：" + " → ".join(r["process"]))
            if r.get("inputs"):
                req_lines.append("**输入数据**：" + "、".join(r["inputs"]))
            if r.get("outputs"):
                req_lines.append("**输出结果**：" + "、".join(r["outputs"]))
        else:
            req_lines.append(f"- {r}")
    req_text = "\n".join(req_lines) or "（请基于访谈记录梳理需求）"

    prompt = f"""根据以下访谈记录和已梳理需求，生成一份专业的IT需求规格说明书（Markdown格式）。

## 访谈记录
{history_text}

## 已梳理需求
{req_text}

请输出包含以下章节的完整文档：
1. **项目背景与目标**
2. **涉及角色与权限**
3. **业务流程说明**
4. **功能需求清单**（每条需求含描述、验收标准）
5. **界面设计要点**（关键页面布局说明）
6. **数据需求**（关键字段、来源、格式、存储要求）
7. **非功能性需求**（性能、安全、合规条款）
8. **待确认事项**（仍需业务方明确的问题）
9. **验收标准**

要求：专业严谨，结构清晰，可直接交付IT团队进行工作量评估。"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        raise HTTPException(500, f"导出失败：{e}")

    return {"document": response.choices[0].message.content}


@app.post("/api/generate")
async def generate_section(req: GenerateRequest):
    history_text = "\n\n".join(
        f"{'业务方' if m.role == 'user' else '分析师'}：{m.content}"
        for m in req.history
    )

    if req.type == "flowchart":
        prompt = f"""根据以下访谈记录，生成一份完整的 Mermaid 业务流程图。

访谈记录：
{history_text}

要求：
- 使用 graph TD 方向
- 添加 classDef 样式（startEnd/process/decision/output/warning）
- 中文标签，每个标签不超过12字
- 使用 subgraph 对阶段分组
- 只输出 Mermaid 代码，不要任何其他文字"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o", max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            raise HTTPException(500, f"生成失败：{e}")
        raw = response.choices[0].message.content
        cleaned = re.sub(r"```(?:mermaid)?\s*", "", raw).replace("```", "").strip()
        return {"type": "flowchart", "mermaid": cleaned, "requirements": []}

    elif req.type == "wireframe":
        # 生成功能流程设计图（Mermaid），侧重系统交互路径与分支方案
        prompt = f"""根据以下访谈记录，生成一份 IT 人员使用的功能流程设计图（Mermaid 格式）。

访谈记录：
{history_text}

要求：
- 使用 graph TD 方向（竖向流程，便于上下阅读）
- 如有多个实现方案或路径，用 subgraph 分别展示（如 subgraph 方案一、subgraph 方案二）
- 矩形节点表示操作步骤，菱形{{}}表示判断节点，圆角([])表示开始/结束
- 中文标签，每个标签不超过10字
- 添加以下 classDef：
  classDef user fill:#fff3cd,stroke:#f59e0b,color:#92400e
  classDef system fill:#dbeafe,stroke:#3b82f6,color:#1e40af
  classDef decision fill:#fce7f3,stroke:#ec4899,color:#831843
  classDef endpoint fill:#d1fae5,stroke:#10b981,color:#065f46
- 用户操作节点加 :::user，系统处理节点加 :::system，判断节点加 :::decision，开始/结束加 :::endpoint
- 只输出 Mermaid 代码，不要任何其他文字"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o", max_tokens=2500,
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            raise HTTPException(500, f"生成失败：{e}")
        raw = response.choices[0].message.content
        cleaned = re.sub(r"```(?:mermaid)?\s*", "", raw).replace("```", "").strip()
        return {"type": "wireframe", "mermaid": cleaned, "requirements": []}

    elif req.type in ("summary", "preview"):
        if req.type == "summary":
            extra = "只需要 module/feature/description/process/inputs/outputs 字段，无需 wireframe。"
        else:
            extra = "必须包含 wireframe 字段，格式：顶部：...\\n中部：...\\n底部：..."

        prompt = f"""根据以下访谈记录，提取并整理所有已明确的功能需求，以 JSON 数组格式输出。

访谈记录：
{history_text}

要求：
- 每个需求对象包含：module（模块名）、feature（功能名）、description（功能说明）、wireframe（界面布局，格式"顶部：...\\n中部：...\\n底部：..."）、process（操作步骤数组）、inputs（输入项数组）、outputs（输出项数组）
- {extra}
- 只输出 JSON 数组，不要任何其他文字
- 确保 JSON 格式合法"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o", max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            raise HTTPException(500, f"生成失败：{e}")
        raw = response.choices[0].message.content
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return {"type": req.type, "requirements": data, "mermaid": ""}
        except Exception:
            match = re.search(r"\[[\s\S]*\]", cleaned)
            if match:
                try:
                    return {"type": req.type, "requirements": json.loads(match.group()), "mermaid": ""}
                except Exception:
                    pass
        return {"type": req.type, "requirements": [], "mermaid": ""}

    raise HTTPException(400, "未知生成类型")


@app.post("/api/analyze-definition")
async def analyze_definition(req: ExportRequest):
    history_text = "\n\n".join(
        f"{'业务方' if m.role == 'user' else '分析师'}：{m.content}"
        for m in req.history
    )
    req_text = "\n".join([
        f"- {r.get('module','')}: {r.get('feature','')} — {r.get('description','')}"
        if isinstance(r, dict) else f"- {r}"
        for r in req.requirements
    ]) if req.requirements else "（暂无结构化需求，请根据访谈记录分析）"

    prompt = f"""你是朝曦金融科技架构顾问，精通 Claude Agent SDK 中 Skill 和 Agent 的设计原则。

请根据以下访谈记录和已梳理需求，判断该需求更适合实现为 Skill 还是 Agent，并给出详细分析。

## 访谈记录
{history_text}

## 已梳理需求
{req_text}

## 判断标准参考

**Skill（专项技能）适合以下情形：**
- 输入输出明确，逻辑相对固定
- 单一职责，完成一件具体的事
- 无需自主决策或动态规划
- 可被其他系统或 Agent 直接调用
- 执行时间短，结果可预期

**Agent（自主智能体）适合以下情形：**
- 需要多步骤推理和动态决策
- 要根据中间结果调整后续行动
- 需要协调调用多个工具或 Skill
- 有较长任务生命周期或需要持久化状态
- 能处理异常情况并自主恢复

请严格按以下 JSON 格式输出，不要输出任何其他文字：
{{
  "recommendation": "Skill" 或 "Agent" 或 "Skill + Agent 组合",
  "confidence": 0到100的整数,
  "summary": "一句话核心结论，不超过40字",
  "skill_fit": 0到100整数,
  "agent_fit": 0到100整数,
  "reasons_for": ["推荐此方案的核心理由1（结合具体需求场景）", "理由2", "理由3"],
  "reasons_against": ["不宜用另一种方式的理由1", "理由2"],
  "skill_roles": ["若涉及Skill，它负责的具体模块1", "模块2"],
  "agent_roles": ["若涉及Agent，它负责的具体能力1", "能力2"],
  "architecture_suggestion": "2-3句话的架构落地建议，结合朝曦业务场景"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o", max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        raise HTTPException(500, f"分析失败：{e}")

    raw = response.choices[0].message.content
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    raise HTTPException(500, "分析结果解析失败，请重试")


@app.post("/api/export/word")
async def export_word(req: WordExportReq):
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    import urllib.parse

    doc = Document()

    # 设置文档默认字体
    style = doc.styles['Normal']
    style.font.name = '微软雅黑'
    style.font.size = Pt(11)

    if req.type == "requirements":
        # 标题
        title = doc.add_heading('需求梳理报告', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph(f'生成时间：{__import__("datetime").datetime.now().strftime("%Y年%m月%d日 %H:%M")}').alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph('')

        for i, r in enumerate(req.requirements):
            if isinstance(r, str):
                doc.add_heading(f'{i+1}. {r}', 2)
                continue
            # 功能标题
            h = doc.add_heading(f'{i+1}. {r.get("module","功能")} · {r.get("feature","")}', 1)
            # 功能说明
            p = doc.add_paragraph()
            p.add_run('功能说明：').bold = True
            p.add_run(r.get('description', ''))
            # 处理流程
            if r.get('process'):
                doc.add_paragraph().add_run('处理流程：').bold = True
                for j, step in enumerate(r['process']):
                    doc.add_paragraph(f'{j+1}. {step}', style='List Number')
            # 输入输出
            if r.get('inputs'):
                p2 = doc.add_paragraph()
                p2.add_run('输入数据：').bold = True
                p2.add_run('、'.join(r['inputs']))
            if r.get('outputs'):
                p3 = doc.add_paragraph()
                p3.add_run('输出结果：').bold = True
                p3.add_run('、'.join(r['outputs']))
            # 界面草图
            if r.get('wireframe'):
                p4 = doc.add_paragraph()
                p4.add_run('界面布局：').bold = True
                p4.add_run(r['wireframe'])
            doc.add_paragraph('─' * 40)

        filename = '需求梳理报告.docx'

    elif req.type == "prd":
        # 解析 markdown 格式的 PRD 文本
        lines = req.prd_text.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                doc.add_paragraph('')
                continue
            if line_stripped.startswith('# '):
                h = doc.add_heading(line_stripped[2:], 0)
                h.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif line_stripped.startswith('## '):
                doc.add_heading(line_stripped[3:], 1)
            elif line_stripped.startswith('### '):
                doc.add_heading(line_stripped[4:], 2)
            elif line_stripped.startswith('#### '):
                doc.add_heading(line_stripped[5:], 3)
            elif line_stripped.startswith(('- ', '* ', '• ')):
                doc.add_paragraph(line_stripped[2:], style='List Bullet')
            elif line_stripped and line_stripped[0].isdigit() and '. ' in line_stripped[:4]:
                doc.add_paragraph(line_stripped, style='List Number')
            elif line_stripped.startswith('**') and line_stripped.endswith('**'):
                p = doc.add_paragraph()
                p.add_run(line_stripped.strip('*')).bold = True
            else:
                # 处理行内加粗 **text**
                p = doc.add_paragraph()
                import re as _re
                parts = _re.split(r'\*\*(.+?)\*\*', line_stripped)
                for k, part in enumerate(parts):
                    run = p.add_run(part)
                    if k % 2 == 1:
                        run.bold = True

        filename = 'PRD需求规格说明书.docx'
    else:
        raise HTTPException(400, "未知导出类型")

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    encoded = urllib.parse.quote(filename)
    return Response(
        content=buffer.getvalue(),
        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        headers={'Content-Disposition': f"attachment; filename*=UTF-8''{encoded}"}
    )


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """接收录音文件，调用 Whisper 转文字"""
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            return {"error": "收到空文件，请重新录音"}

        fname = file.filename or "audio.webm"
        content_type = file.content_type or "audio/webm"
        print(f"[transcribe] 收到文件: {fname}, 类型: {content_type}, 大小: {len(audio_bytes)} bytes")

        # 尝试调用 Whisper
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=(fname, audio_bytes, content_type),
            language="zh"
        )
        text = transcript.text.strip()
        print(f"[transcribe] 识别结果: {text[:80]}")
        return {"text": text}

    except Exception as e:
        err_msg = str(e)
        print(f"[transcribe] 错误: {err_msg}")
        # 返回 200 + error 字段，前端可以显示具体原因
        return {"text": "", "error": f"识别失败：{err_msg[:200]}"}


@app.post("/api/generate-demo")
async def generate_demo(req: DemoRequest):
    """根据需求访谈内容生成可交互的HTML原型Demo"""
    history_text = "\n".join(
        f"{'用户' if m.role=='user' else 'AI分析师'}: {m.content[:300]}"
        for m in req.history[-30:]
    )
    req_text = ""
    if req.requirements:
        items = req.requirements if isinstance(req.requirements[0], str) else [
            f"- {r.get('feature','')}: {r.get('description','')}" for r in req.requirements
        ]
        req_text = "\n".join(items[:15])

    prompt = f"""你是一位资深前端工程师，擅长快速制作高保真交互原型。

【需求访谈记录】
{history_text}

【整理后的功能需求】
{req_text if req_text else '（请从访谈记录中提取）'}

请根据以上需求，生成一个完整的、可交互的HTML原型Demo，要求：

1. **完全自包含**：单个HTML文件，内联CSS和JavaScript，不依赖任何外部CDN或资源
2. **真实可用**：包含与需求匹配的模拟数据（3-8条示例记录），用户可以真实操作（点击、筛选、填写表单、查看详情等）
3. **专业外观**：金融/企业级UI风格，配色深蓝+白，简洁专业；使用系统字体
4. **核心流程完整**：覆盖需求中最关键的1-2个主流程，让用户能感受到产品的核心价值
5. **导航清晰**：如有多个功能模块，用标签页或侧边栏组织，每个模块都可点击操作
6. **数据互动**：支持增删改查中至少2种操作（如新增记录、筛选查询、查看详情、状态变更等）
7. **朝曦场景**：数据内容贴合家族财富/税务/资本市场/法律/企业治理等专业场景

注意：
- 不要使用 alert()，改用页内提示
- 表格数据要真实，字段名称要专业
- 按钮点击要有反馈（高亮、状态变化、列表刷新等）
- 顶部显示"⚡ 需求原型Demo — [系统名称]（仅供需求确认，非最终产品）"

只输出完整的HTML代码，从<!DOCTYPE html>开始，不要有任何解释文字。"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是专业的前端原型工程师，直接输出完整HTML代码，不含任何Markdown标记或解释。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=6000,
            temperature=0.4
        )
        html = resp.choices[0].message.content.strip()
        # 清理可能的 markdown 代码块包裹
        if html.startswith("```"):
            html = html.split("```", 2)[1]
            if html.startswith("html"):
                html = html[4:]
            html = html.rsplit("```", 1)[0].strip()
        # 校验是否包含有效 HTML
        if "<html" not in html.lower() and "<!doctype" not in html.lower():
            raise HTTPException(500, f"API返回内容不是有效HTML，可能被截断。原始内容前200字：{html[:200]}")
        return {"html": html}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Demo生成失败：{type(e).__name__}: {str(e)[:300]}")


# ─────────────────────────────────────────────
# 用户认证 & 会话云端存储
# ─────────────────────────────────────────────

@app.post("/api/auth/login")
async def login(req: LoginRequest):
    """邮箱登录（无密码，内部工具）。首次自动注册，老用户直接返回 token。"""
    email = req.email.strip().lower()
    name  = req.name.strip()
    if not email or "@" not in email:
        raise HTTPException(400, "请输入有效的邮箱地址")
    if not name:
        raise HTTPException(400, "请输入您的姓名")

    admin_email = os.environ.get("ADMIN_EMAIL", "").strip().lower()
    is_admin = 1 if email == admin_email else 0

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        now = datetime.now().isoformat()
        if user:
            new_admin = max(is_admin, int(user["is_admin"]))
            cur.execute("UPDATE users SET name=%s, last_active=%s, is_admin=%s WHERE email=%s",
                        (name, now, new_admin, email))
            conn.commit()
            return {"token": user["token"], "user_id": user["id"],
                    "user": {"name": name, "email": email, "is_admin": bool(new_admin)}}
        else:
            uid   = secrets.token_hex(8)
            token = secrets.token_hex(24)
            cur.execute("INSERT INTO users VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (uid, email, name, token, is_admin, now, now))
            conn.commit()
            return {"token": token, "user_id": uid,
                    "user": {"name": name, "email": email, "is_admin": bool(is_admin)}}
    finally:
        cur.close(); conn.close()

@app.get("/api/auth/me")
async def get_me(authorization: Optional[str] = Header(None)):
    user = require_token(authorization)
    return {"user_id": user["id"], "name": user["name"],
            "email": user["email"], "is_admin": bool(user["is_admin"])}

@app.post("/api/sessions/save")
async def save_session(req: SaveSessionRequest, authorization: Optional[str] = Header(None)):
    """保存或更新一条会话到服务端"""
    user = require_token(authorization)
    now  = datetime.now().isoformat()
    conn = get_db()
    cur  = conn.cursor()
    try:
        sid = req.session_id or secrets.token_hex(10)
        cur.execute("SELECT id FROM sessions WHERE id=%s AND user_id=%s", (sid, user["id"]))
        existing = cur.fetchone()
        h_json = json.dumps(req.history, ensure_ascii=False)
        r_json = json.dumps(req.requirements, ensure_ascii=False)
        if existing:
            cur.execute("""UPDATE sessions SET history=%s,requirements=%s,mermaid=%s,
                           completeness=%s,title=%s,updated_at=%s WHERE id=%s""",
                        (h_json, r_json, req.mermaid, req.completeness, req.title, now, sid))
        else:
            cur.execute("""INSERT INTO sessions VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (sid, user["id"], req.title, h_json, r_json,
                         req.mermaid, req.completeness, now, now))
        conn.commit()
        return {"session_id": sid, "updated_at": now}
    finally:
        cur.close(); conn.close()

@app.get("/api/sessions")
async def list_sessions(authorization: Optional[str] = Header(None)):
    """获取当前用户的所有会话列表（不含完整 history）"""
    user = require_token(authorization)
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("""SELECT id,title,completeness,created_at,updated_at
                       FROM sessions WHERE user_id=%s ORDER BY updated_at DESC""",
                    (user["id"],))
        return {"sessions": [dict(r) for r in cur.fetchall()]}
    finally:
        cur.close(); conn.close()

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str, authorization: Optional[str] = Header(None)):
    """获取单条会话完整内容"""
    user = require_token(authorization)
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT * FROM sessions WHERE id=%s AND user_id=%s",
                    (session_id, user["id"]))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "会话不存在")
        d = dict(row)
        d["history"]      = json.loads(d["history"] or "[]")
        d["requirements"] = json.loads(d["requirements"] or "[]")
        return d
    finally:
        cur.close(); conn.close()

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, authorization: Optional[str] = Header(None)):
    user = require_token(authorization)
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("DELETE FROM sessions WHERE id=%s AND user_id=%s",
                    (session_id, user["id"]))
        conn.commit()
        return {"ok": True}
    finally:
        cur.close(); conn.close()

# ─────────────────────────────────────────────
# 管理员看板
# ─────────────────────────────────────────────

@app.get("/api/admin/stats")
async def admin_stats(authorization: Optional[str] = Header(None)):
    require_admin(authorization)
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM users"); users = cur.fetchone()["count"]
        cur.execute("SELECT COUNT(*) FROM sessions"); sessions = cur.fetchone()["count"]
        cur.execute("""SELECT COUNT(DISTINCT user_id) FROM sessions
                       WHERE updated_at > (NOW() - INTERVAL '7 days')::text""")
        active = cur.fetchone()["count"]
        return {"total_users": users, "total_sessions": sessions, "active_7d": active}
    finally:
        cur.close(); conn.close()

@app.get("/api/admin/users")
async def admin_users(authorization: Optional[str] = Header(None)):
    require_admin(authorization)
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("""
            SELECT u.id, u.name, u.email, u.created_at, u.last_active,
                   COUNT(s.id) as session_count,
                   MAX(s.updated_at) as last_session,
                   MAX(s.completeness) as max_completeness
            FROM users u
            LEFT JOIN sessions s ON s.user_id = u.id
            WHERE u.is_admin = 0
            GROUP BY u.id ORDER BY u.last_active DESC
        """)
        return {"users": [dict(r) for r in cur.fetchall()]}
    finally:
        cur.close(); conn.close()

@app.get("/api/admin/users/{user_id}/sessions")
async def admin_user_sessions(user_id: str, authorization: Optional[str] = Header(None)):
    require_admin(authorization)
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("""SELECT id,title,completeness,created_at,updated_at
                       FROM sessions WHERE user_id=%s ORDER BY updated_at DESC""",
                    (user_id,))
        return {"sessions": [dict(r) for r in cur.fetchall()]}
    finally:
        cur.close(); conn.close()

@app.get("/api/admin/sessions/{session_id}")
async def admin_get_session(session_id: str, authorization: Optional[str] = Header(None)):
    require_admin(authorization)
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT * FROM sessions WHERE id=%s", (session_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "会话不存在")
        d = dict(row)
        d["history"]      = json.loads(d["history"] or "[]")
        d["requirements"] = json.loads(d["requirements"] or "[]")
        return d
    finally:
        cur.close(); conn.close()

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """管理员看板页面"""
    return HTMLResponse(open("static/admin.html", encoding="utf-8").read()
                        if os.path.exists("static/admin.html") else "<h1>管理页面未找到</h1>")


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
