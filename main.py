import json
import re
import io
import os
from typing import List, Union

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
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


class ChatRequest(BaseModel):
    history: List[Message]
    message: str


class ExportRequest(BaseModel):
    history: List[Message]
    requirements: List[Union[dict, str]]
    mermaid: str


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
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {
        "message": raw,
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


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
