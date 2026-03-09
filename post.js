function slugifyHeading(text) {
  return text
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .trim()
    .replace(/[^\w\u4e00-\u9fff\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
}

function assignHeadingIds(headings) {
  const seenIds = new Map();

  headings.forEach((heading, index) => {
    const source = heading.id || heading.textContent || `section-${index + 1}`;
    const baseId = slugifyHeading(source) || `section-${index + 1}`;
    const repeat = seenIds.get(baseId) || 0;
    const nextId = repeat === 0 ? baseId : `${baseId}-${repeat + 1}`;

    seenIds.set(baseId, repeat + 1);
    heading.id = nextId;
  });
}

function buildOutline() {
  const article = document.getElementById("post-content");
  const outline = document.getElementById("post-outline");
  const nav = document.getElementById("post-outline-nav");

  if (!article || !outline || !nav) {
    return;
  }

  const headings = [...article.querySelectorAll("h2, h3, h4")];

  if (headings.length === 0) {
    return;
  }

  assignHeadingIds(headings);

  const fragment = document.createDocumentFragment();

  headings.forEach((heading) => {
    const link = document.createElement("a");
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent.trim();
    link.className = `toc-link toc-${heading.tagName.toLowerCase()}`;
    link.dataset.target = heading.id;
    fragment.appendChild(link);
  });

  nav.replaceChildren(fragment);
  outline.hidden = false;

  const links = [...nav.querySelectorAll(".toc-link")];
  const activeLinkById = new Map(links.map((link) => [link.dataset.target, link]));

  const setActive = (id) => {
    links.forEach((link) => {
      link.classList.toggle("active", link.dataset.target === id);
    });
  };

  if ("IntersectionObserver" in window) {
    const visibleIds = new Set();
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            visibleIds.add(entry.target.id);
          } else {
            visibleIds.delete(entry.target.id);
          }
        });

        const current = headings.find((heading) => visibleIds.has(heading.id));
        if (current && activeLinkById.has(current.id)) {
          setActive(current.id);
        }
      },
      {
        rootMargin: "0px 0px -70% 0px",
        threshold: 0.1
      }
    );

    headings.forEach((heading) => observer.observe(heading));
  }

  const firstHeading = headings[0];
  if (firstHeading) {
    setActive(firstHeading.id);
  }
}

function watchPageViews() {
  const viewContainer = document.getElementById("busuanzi_container_page_pv");
  const viewValue = document.getElementById("busuanzi_value_page_pv");

  if (!viewContainer || !viewValue) {
    return;
  }

  window.setTimeout(() => {
    if (!viewValue.textContent.trim()) {
      viewContainer.hidden = true;
    }
  }, 2500);
}

document.addEventListener("DOMContentLoaded", () => {
  buildOutline();
  watchPageViews();
});
