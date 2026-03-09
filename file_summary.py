def load_parent_ids_from_yaml(yaml_path):
    p = Path(yaml_path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    node = data.get("parent_id")
    parent_ids = []
    if isinstance(node, list):
        parent_ids = [str(x) for x in node if x is not None]
    elif isinstance(node, dict):
        def sort_key(k):
            try:
                return (0, int(k))
            except ValueError:
                return (1, str(k))
        for k in sorted(node.keys(), key=sort_key):
            v = node[k]
            if v is not None:
                parent_ids.append(str(v))
    else:
        parent_ids = []
    return parent_ids

def get_page_title_by_id(page_id):
    if not page_id:
        return None
    page_id = get_id(page_id)
    if not page_id:
        return None
    api_url = f"{CONFLUENCE_BASE_URL}/rest/api/content/{page_id}"
    try:
        resp = session.get(api_url)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("title")
        if resp.status_code == 404:
            return None
        return None
    except Exception as exc:
        print(f"Error connecting to API: {exc}")
        return None

def list_child_pages(parent_id):
    resolved_id = get_id(parent_id)
    if not resolved_id:
        return []
    children = []
    start = 0
    limit = 50
    while Tur:
        api_url = f"{CONFLUENCE_BASE_URL}/rest/api/content/{resolved_id}/child/page"
        params = {"start": start, "limit": limit}
        resp = session.get(api_url, params=params)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        children.extend(results)
        if data.get("_links", {}).get("next"):
            start += limit
            continue
        break
    return children

def build_child_page_rows(children):
    if not children:
        return []
    def is_summary_page(title: str) -> bool:
        return title.strip().endswith("summary")
    rows = []
    for index, page in enumerate(children, start=1):
        title = page.get("title", "")
        if is_summary_page(title):
            continue
        page_id = page.get("id", "")
        if has_child_pages(page_id):
            rows.append((index, title, page_id, "-", "-", "", None, None, None, None, "-"))
            continue
        toplevel_comments = list_top_level_comments(page_id)
        if not toplevel_comments:
            rows.append((index, title, page_id, "-", "-", "", None, None, None, None, "-"))
            continue
        for root_comment in toplevel_comments:
            thread_comments = collect_comment_thread(root_comment)
            value = root_comment.get("body", {}).get("storage", {}).get("value", "")
            cleaned = build_thread_comment_text(thread_comments)
            author_name = get_comment_author_name(root_comment)
            comment_link = get_comment_link(root_comment, page_id)
            comment_id = root_comment.get("id")
            row_anchor = build_row_anchor(comment_id) if comment_id else None
            comment_version = root_comment.get("version", {}).get("number")
            resolution_status = get_comment_resolution_status(root_comment)
            rows.append((
                index,
                title,
                page_id,
                cleaned,
                author_name,
                comment_link,
                comment_id,
                row_anchor,
                value,
                comment_version,
                resolution_status
            ))
    return rows


def get_parent_context(parent_id):
    resolved_id = get_id(parent_id)
    if not resolved_id:
        return None
    api_url = f"{CONFLUENCE_BASE_URL}/rest/api/content/{resolved_id}"
    resp = session.get(api_url, params = {"expand": "space"})
    resp.raise_for_status()
    data = resp.json()
    space_key = data.get("space", {}).get("key")
    parent_title = data.get("title")
    target_parent_id = resolved_id
    return {
        "space_key": space_key,
        "parent_title": parent_title,
        "target_parent_id": target_parent_id
    }

def build_storage_table(rows):
    header = (
        "<table>"
        "<thead><tr>"
        "<th>#</th><th>Page File</th><th>Page ID</th><th>Comment</th><th>Comment Author</th><th>Status</th>"
        "</tr></thead>"
        "<tbody>"
    )
    body_parts = []
    for row in rows:
        comment_xhtml = row[3].replace("\n", "<br/>")
        comment_link = row[5]
        row_anchor = row[7]

        if comment_xhtml != "-" and comment_link:
            comment_cell = (
                f'<a href="{html.escape(comment_link, quote=True)}" >link to file</a>'
                f"{comment_xhtml}"
            )
        else:
            comment_cell = "-"
        tr_open = f'<tr id="{html.escape(row_anchor, quote=True)}" >' if row_anchor else "<tr>"
        body_parts.append(
            tr_open
            + f"<td>{html.escape(str(row[0]))}</td>"
            + f"<td>{html.escape(str(row[1]))}</td>"
            + f"<td>{html.escape(str(row[2]))}</td>"
            + f"<td>{comment_cell}</td>"
            + f"<td>{html.escape(str(row[4]))}</td>"
            + f"<td>{html.escape(str(row[10]))}</td>"
            + "</tr>"
        )
    return header + "".join(body_parts) + "</tbody></table>"

def upsert_summary_page(parent_context, rows):
    if not parent_context:
        return None
    if not rows:
        rows = [("-", "-", "-", "-", "-", "-", None, None, None, None, "-")]
    summary_title = f"{parent_context['parent_title']} summary"
    storage_vale = build_storage_table(rows)
    existing_pae = find_existing_summary_page(parent_context["target_parent_id"], summary_title, parent_context["space_key"])
    if existing_pae:
        return update_summary_page(
            existing_pae,
            summary_title,
            storage_vale,
            parent_context["target_parent_id"],
        )
    payload = {
        "type": "page",
        "title": summary_title,
        "space": {"key": parent_context["space_key"]},
        "body": {
            "storage": {
                "value": storage_vale,
                "representation": "storage"
            }
        }
    }
    if parent_context["target_parent_id"]:
        payload["ancestors"] = [{"id": parent_context["target_parent_id"]}]
    api_url = f"{CONFLUENCE_BASE_URL}/rest/api/content/"
    resp = session.post(api_url, json=payload)
    resp.raise_for_status()
    page_id = resp.json().get("id")
    return page_id

def main():
    parser = argparse.ArgumentParser(description="Update Confluence landing page with latest version info.")
    args = parser.parse_args()
    all_parent_ids = []
    for y in args.yaml:
        ids = load_parent_ids_from_yaml(y)
        if not ids:
            print(f"Warning: No parent IDs found in {y}")
            continue
        all_parent_ids.extend(ids)
    if not all_parent_ids:
        return 

    # 依次处理每一个parent_id
    for parent_id in all_parent_ids:
        parent_title = get_page_title_by_id(parent_id)
        if not parent_title:
            print(f"Warning: Parent page with ID {parent_id} not found. Skipping.")
            continue
        children = list_child_pages(parent_id)
        rows = build_child_page_rows(children)
        parent_context = get_parent_context(parent_id)
        summary_page_id = upsert_summary_page(parent_context, rows)
        update_comment_backlinks(rows, summary_page_id)