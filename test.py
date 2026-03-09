import argparse
import base64
import os
import re
import subprocess
import sys
import tempfile

import requests  # type: ignore
import yaml  # type: ignore

BASE_URL = os.environ.get("CONFLUENCE_URL")
TOKEN = os.environ.get("CONFLUENCE_TOKEN")
BR_TAG_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<!--.*?-->|</?[A-Za-z][^>]*?>", re.DOTALL)

if not BASE_URL:
    BASE_URL = "https://wiki.infiscale.dev/"

session = requests.Session()

session.headers.update({
    "Authorization": f"Bearer {TOKEN}",
})

if not all([BASE_URL, TOKEN]):
    print("Error: 环境变量 CONFLUENCE_TOKEN 必须设置")
    sys.exit(1)


def get_id(init_id):
    tiny_id = init_id.split("/")[-1]
    try:
        padding = len(tiny_id) % 4
        if padding > 0:
            tiny_id += "=" * (4 - padding)
        tiny_id = tiny_id.replace("-", "+").replace("_", "/")
        decoded_bytes = base64.b64decode(tiny_id)

        page_id = int.from_bytes(decoded_bytes, byteorder="little")
        print(f"Real ID of {init_id}: {page_id}")
        return str(page_id)
    except Exception as exc:
        print(f"Error: {exc}")
        return None


def get_page_title_by_id(page_id):
    """
    关键函数：通过 ID 查询页面的当前标题
    """
    if not page_id:
        return None

    page_id = get_id(page_id)
    if not page_id:
        return None

    api_url = f"{BASE_URL}rest/api/content/{page_id}"
    try:
        resp = session.get(api_url)
        if resp.status_code == 200:
            return resp.json().get("title")
        if resp.status_code == 404:
            print(f"Error: Parent ID {page_id} 不存在！")
            return None
        print(f"Error: 查询页面失败 {resp.status_code}: {resp.text}")
        return None
    except Exception as exc:
        print(f"Error connecting to API: {exc}")
        return None



def normalize_br_tags(text):
    """
    统一各种 br 写法：
    <br> / <br/> / <br />  -> <br />
    """
    return BR_TAG_RE.sub("<br />", text)


def protect_br_tags(text):
    """
    先把 <br /> 替换成占位符，避免后续被误伤
    """
    placeholders = []

    def repl(_match):
        idx = len(placeholders)
        placeholders.append("<br />")
        return f"__MARK_BR_{idx}__"

    text = normalize_br_tags(text)
    text = BR_TAG_RE.sub(repl, text)
    return text, placeholders


def restore_placeholders(text, placeholders):
    for i, value in enumerate(placeholders):
        text = text.replace(f"__MARK_BR_{i}__", value)
    return text


def break_unsupported_html(text):
    """
    保留 HTML comments 避免破坏 mark 的 metadata / macro comment
    其他 HTML tag 继续按保底逻辑处理
    """
    def repl(match):
        token = match.group(0)

        if token.startswith("<!--"):
            return token

        return token.replace("<", "< ")

    return HTML_TAG_RE.sub(repl, text)


def preprocess_non_code_text(text):
    """
    非 code fence 区域：
    - 保护 <br>
    - 破坏其他 HTML
    - 恢复 <br>
    """
    text, placeholders = protect_br_tags(text)
    text = break_unsupported_html(text)
    text = restore_placeholders(text, placeholders)
    return text


def preprocess_content_preserve_fences(content):
    """
    只处理非 fenced code block 的内容
    fenced code block 内完全不动
    """
    lines = content.splitlines(keepends=True)
    out = []
    buf = []

    in_fence = False
    fence_char = ""
    fence_len = 0

    def flush_buf():
        nonlocal buf
        if buf:
            out.append(preprocess_non_code_text("".join(buf)))
            buf = []

    for line in lines:
        fence = is_fence_line(line)

        if not in_fence and fence:
            flush_buf()
            in_fence = True
            fence_char, fence_len = fence
            out.append(line)
            continue

        if in_fence:
            out.append(line)
            if is_fence_close(line, fence_char, fence_len):
                in_fence = False
            continue

        buf.append(line)

    flush_buf()
    return "".join(out)

def is_fence_line(line):
    trimmed = line.lstrip()
    while trimmed.startswith(">"):
        trimmed = trimmed[1:].lstrip()

    if len(trimmed) < 3:
        return None

    fence_char = trimmed[0]
    if fence_char not in ("`", "~"):
        return None

    count = 0
    while count < len(trimmed) and trimmed[count] == fence_char:
        count += 1

    if count < 3:
        return None

    return fence_char, count


def is_fence_close(line, fence_char, fence_len):
    trimmed = line.lstrip()
    while trimmed.startswith(">"):
        trimmed = trimmed[1:].lstrip()

    if len(trimmed) < fence_len:
        return False

    count = 0
    while count < len(trimmed) and trimmed[count] == fence_char:
        count += 1

    if count < fence_len:
        return False

    return trimmed[count:].strip() == ""




def generate_mark_config(yaml_config_path, tag):
    """
    read yaml, generate data structure for mark tools
    """
    with open(yaml_config_path, "r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)

    default_space = config_data.get("default_space")

    pages_to_sync = []

    print(f"--- Preparing Sync for Tag: {tag} ---")
    print(config_data)
    for rule in config_data["rules"]:
        space = rule.get("space", default_space)
        source_path = rule["path"]
        parent_id = rule.get("parent_id")

        if not os.path.exists(source_path):
            print(f"Checking: {source_path} ... NOT FOUND (Skipping)")
            continue

        parent_title = get_page_title_by_id(parent_id)
        if not parent_title:
            print(f"Checking: Page {parent_id} ... NOT FOUND (Skipping)")
            continue

        page_entry = {
            "path": source_path,
            "space": space,
            "parent": parent_title,
        }
        pages_to_sync.append(page_entry)

    if not pages_to_sync:
        print("No new pages to publish.")

    return pages_to_sync


def append_version(page, tag):
    path = page["path"]
    dir_path = os.path.dirname(path)
    filename = os.path.basename(path)
    filename = os.path.splitext(filename)[0]  # 去掉 .md

    version = f"<!-- Title: {filename}_{tag} -->\n"
    with open(path, "r", encoding="utf-8") as handle:
        original_content = handle.read()

    original_content = "#\n" + preprocess_content_preserve_fences(original_content)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        dir=dir_path,
        delete=False,
        encoding="utf-8",
    ) as temp_file:
        temp_file.write(version + original_content)
        temp_file_path = temp_file.name

    page["path"] = temp_file_path
    page["title"] = f"{filename}_{tag}"
    return page, filename


def run_mark_tool(page):
    """调用 mark 二进制文件执行同步"""
    print("\n--- Running mark ---")
    try:
        subprocess.run(
            [
                "mark",
                # "--dry-run",
                "-b",
                BASE_URL,
                "-p",
                TOKEN,
                "-f",
                page["path"],
                "--space",
                page["space"],
                "--parents",
                page["parent"],
                "--features",
                "mkdocsadmonitions",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Error running mark tool: {exc}")
        sys.exit(1)
    finally:
        if os.path.exists(page["path"]):
            # pass
            os.remove(page["path"])


def update_landing_page(page, newpage):
    title = page["parent"]
    storage_value = (
        f'<!-- Title: {title} -->\n\n'
        "<strong>Current Latest Version:</strong> "
        f'<ac:link><ri:page ri:content-title="{newpage["title"]}" '
        f'ri:space-key="{newpage["space"]}"/></ac:link>\n'
        '<ac:structured-macro ac:name="children" ac:schema-version="1">\n'
        '<ac:parameter ac:name="sort">creation</ac:parameter>\n'
        '<ac:parameter ac:name="reverse">true</ac:parameter>\n'
        "</ac:structured-macro>\n"
    )

    path = page["path"]
    dir_path = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        dir=dir_path,
        delete=False,
        encoding="utf-8",
    ) as temp_file:
        temp_file.write(storage_value)
        temp_file_path = temp_file.name

    page["path"] = temp_file_path
    page["parent"] = ""
    return page


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    pages_to_sync = generate_mark_config(args.config, args.tag)
    if pages_to_sync:
        for page in pages_to_sync:
            newpage, _title = append_version(page, args.tag)
            run_mark_tool(newpage)
            landing_page = update_landing_page(page, newpage)
            run_mark_tool(landing_page)
    else:
        print("Skipped execution.")


if __name__ == "__main__":
    main()