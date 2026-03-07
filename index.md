---
layout: default
---

<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Qihang Wu</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap"
      rel="stylesheet"
    />
    <link rel="stylesheet" href="style.css" />
  </head>
  <body>
    <div id="intro" aria-live="polite">
      <div class="intro-shell">
        <h1 id="quote"></h1>
      </div>
    </div>

    <main id="home" class="hidden" aria-hidden="true">
      <header class="topbar">
        <div class="topbar-links social-links" aria-label="Social links">
          <a href="mailto:qihang@example.com">mail</a>
          <a href="https://github.com/" target="_blank" rel="noreferrer">github</a>
          <a href="https://www.linkedin.com/" target="_blank" rel="noreferrer">linkedin</a>
          <a href="#posts">rss</a>
          <a href="#contact">wechat</a>
        </div>
        <nav class="topbar-links site-nav" aria-label="Primary navigation">
          <a class="active" href="#about">about</a>
          <a href="#posts">blog</a>
          <a href="#publications">publications</a>
          <a href="#contact">cv</a>
        </nav>
      </header>

      <section class="hero-layout" id="about">
        <div class="hero-copy">
          <h1 class="hero-title">
            <span>Qihang</span>
            <span>Wu</span>
          </h1>
          <p class="hero-summary">
            <strong>Qihang Wu</strong> is a training and inference framework engineer.
            He is a member of <a href="https://ahu-spvl.github.io/">AHU-SPVL</a>,
            where his earlier research focused on domain adaptation for semantic
            segmentation. He is now working on training and inference systems for
            LLMs and VLMs.
          </p>

          <div class="feature-links" aria-label="Selected links">
            <a href="#publications">
              <strong>CroPe</strong>
              <span>CroPe: Cross-Modal Semantic Compensation
Adaptation for All Adverse Scene Understanding.</span>
            </a>
            <a href="#publications">
              <strong>UCDS</strong>
              <span>Unlocking Cross-Domain Synergies for Domain
Adaptive Semantic Segmentation.</span>
            </a>
            <a href="#posts">
              <strong>Linear Attention</strong>
              <span>Review the development of linear attention and study its hardware-efficient implementation.</span>
            </a>
           
          </div>
        </div>

        <aside class="hero-visual" aria-label="Profile panel">
          <div class="portrait-card">
            <div class="portrait-glow"></div>
            <div class="portrait-frame">
              <div class="portrait-photo" aria-hidden="true">
                <img src="img/me.jpg" alt="Profile photo of Qihang Wu">
              </div>
            </div>
          </div>
        </aside>
      </section>

      <section class="content-section" id="posts">
        <h2>latest posts</h2>
        <div class="list-table">
          {% for post in site.posts limit:3 %}
          <a class="list-row" href="{{ post.url }}">
            <span class="list-date">{{ post.date | date: "%b %d, %Y" }}</span>
            <span class="list-title">{{ post.title }}</span>
          </a>
          {% endfor %}
        </div>
      </section>

      <section class="content-section" id="publications">
        <h2>selected publications</h2>
        <div class="publication-list">
          <article class="publication-item">
            <h3>Scaling Efficient Attention for Practical Training Systems</h3>
            <p>
              Qihang Wu, Collaborators. A systems-oriented study of efficient attention
              kernels for modern large-model training and inference.
            </p>
          </article>
          <article class="publication-item">
            <h3>Domain Adaptation for Semantic Segmentation in Real-World Scenes</h3>
            <p>
              Qihang Wu, Collaborators. Research on robust segmentation transfer under
              cross-domain shifts and limited target supervision.
            </p>
          </article>
        </div>
      </section>

      <footer class="footer-band" id="contact">
        <p>Qihang Wu</p>
        <p>Training and inference framework engineer.</p>
      </footer>
    </main>

    <script src="script.js"></script>
  </body>
</html>
