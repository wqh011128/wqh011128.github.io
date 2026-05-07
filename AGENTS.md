# AGENTS.md

This file is the working guide for Codex agents editing this repository.

## Project Overview

This repository is a personal GitHub Pages site for Qihang Wu. It is a mostly static Jekyll site with hand-authored HTML, one global stylesheet, Markdown posts, and a small amount of JavaScript for blog/post behavior.

Primary goals of the site:

- Present Qihang Wu's identity as an AI systems / algorithm engineer.
- Keep the homepage focused and concise.
- Put deeper technical writing, PDFs, and slides in the resource archive.
- Maintain a polished, readable blog for long technical notes.

## Repository Structure

```text
.
├── index.html          # Homepage. Main target for visual redesign work.
├── blog.html           # Resource/archive page for posts, PDFs, and PPT/PPTX files.
├── style.css           # Global styles for homepage, archive, and posts.
├── post.js             # Post-page table of contents and reading UI behavior.
├── script.js           # Older intro/quote animation code. Currently not central to homepage.
├── _config.yml         # Jekyll site metadata and GitHub Pages settings.
├── _layouts/
│   ├── default.html    # Minimal default layout.
│   └── post.html       # Blog post layout, MathJax config, TOC sidebar.
├── _posts/             # Markdown blog posts. Filenames follow YYYY-MM-DD-title.md.
├── img/                # Image assets used by homepage and posts.
├── pdf/                # Static PDF resources shown by the archive page.
├── ppt/                # Static PPT/PPTX resources shown by the archive page.
├── skills/             # Local Codex skills directory; ignored by git.
├── Gemfile             # Ruby/Jekyll dependency entrypoint.
└── README.md           # Short repository note.
```

## Important Files

### `index.html`

The homepage is hand-written HTML with Jekyll front matter at the top.

Current homepage sections:

- `home-nav`: top navigation and identity mark.
- `home-hero`: first viewport introduction, profile image, primary actions.
- `home-snapshot`: compact signal cards for focus, research, and archive.
- `work`: selected work/project cards.
- `notes`: recent/important writing and archive entry points.
- Footer/contact area if present.

When redesigning the homepage, prefer editing `index.html` and `style.css` together.

### `style.css`

This is the single global stylesheet. It contains styles for:

- Homepage classes, mostly prefixed with `home-`.
- Archive/resource page classes, often prefixed with `archive-` or `resource-`.
- Post layout and Markdown rendering classes, mostly prefixed with `post-`.
- Responsive rules near the bottom.

Be careful when editing shared selectors such as `body`, `a`, `h1`, `main`, `.topbar`, `.post-content`, and media queries. They can affect multiple pages.

### `blog.html`

This page uses Jekyll/Liquid to collect:

- Markdown posts from `site.posts`.
- PPT/PPTX files from `site.static_files`.
- PDF files from `site.static_files`.

Do not replace Liquid expressions with hard-coded lists unless the user explicitly asks for a static archive.

### `_layouts/post.html`

This layout controls individual blog posts. It includes:

- MathJax configuration.
- A helper that normalizes standalone `$$` math blocks.
- Post metadata.
- Sticky generated table of contents.
- Previous/next navigation.

When changing math rendering, check both Markdown syntax in `_posts` and MathJax settings here.

### `_posts/`

Posts are Markdown files with YAML front matter:

```yaml
---
layout: post
title: "Post Title"
date: YYYY-MM-DD
categories:
  - blog
---
```

For formulas, prefer:

- Inline math: `$...$`
- Display math:

```text
$$
...
$$
```

For special model tokens such as `<|image_pad|>` or `<think>`, wrap them in inline code or fenced code blocks so Jekyll/HTML does not interpret them as tags.

## Design Direction

The site should feel like a focused personal technical portfolio, not a dense resume.

Homepage design principles:

- Keep first-screen copy short and memorable.
- Let the profile image, selected work, and writing archive carry credibility.
- Prefer strong visual hierarchy over many explanatory paragraphs.
- Use the homepage as a gateway; put depth in `blog.html` and `_posts`.
- Avoid duplicating the same claims across hero, cards, experience, and footer.

Visual tone:

- Technical, calm, and intentional.
- More editorial than corporate.
- More systems-minded than generic portfolio template.
- Avoid adding decorative elements that do not clarify identity or navigation.

## Frontend Editing Guidelines

- Preserve the existing Jekyll/Liquid setup.
- Keep homepage-specific CSS under `home-` class names when possible.
- Avoid broad selector changes unless intentionally updating the whole site.
- Check mobile layouts after changing grids, hero content, or card text.
- Keep card copy short enough to scan.
- Do not put large amounts of text inside the first viewport.
- Use existing fonts unless doing a deliberate visual system redesign.
- If adding images, put assets in `img/` and reference them through `{{ '/img/file.ext' | relative_url }}` in Liquid-aware HTML files.

## Blog And Markdown Guidelines

- Keep long technical explanations in posts rather than homepage sections.
- Use fenced code blocks with language tags when possible.
- Prefer code blocks for raw prompts, model tokens, shell output, and pseudo code.
- Do not use raw HTML for layout inside Markdown unless Markdown cannot express the structure cleanly.
- Avoid broken exported citation markers such as `turn0search...` or special research-export tokens.
- Keep formulas compatible with MathJax.

## Git And Deployment Notes

Remote repository:

```text
git@github.com:wqh011128/wqh011128.github.io.git
```

Main branch:

```text
main
```

GitHub Pages is expected to build from this repository. Avoid committing local-only agent assets from `skills/`; it is ignored by `.gitignore`.

If Git reports a dubious ownership error, the local machine may need:

```powershell
git config --global --add safe.directory E:/my_github_page/wqh011128.github.io
```

## Verification

Useful checks after edits:

```powershell
git status --short
```

For Markdown/post changes, inspect the affected post in `_posts/` and check that:

- Front matter exists.
- Math uses `$...$` or `$$...$$`.
- Raw `<...>` tokens are escaped with code formatting.

For homepage redesigns, manually review:

- Desktop hero layout.
- Mobile hero layout.
- Navigation links.
- Archive link.
- Profile image path.
- Project links.

## Collaboration Notes

Before a major homepage redesign, first read:

- `index.html`
- `style.css`
- `blog.html`
- `_layouts/post.html`

Then identify whether the change is:

- Content architecture: mostly `index.html`.
- Visual system: mostly `style.css`.
- Blog/archive behavior: `blog.html`, `_layouts/post.html`, and `post.js`.
- Long-form technical content: `_posts/`.

Keep changes scoped and explain the design reasoning in the final response.
