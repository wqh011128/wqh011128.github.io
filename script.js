const quote = "Lose yourself to find yourself";
const intro = document.getElementById("intro");
const quoteEl = document.getElementById("quote");
const home = document.getElementById("home");

function buildQuote() {
  const words = quote.split(" ");
  const fragment = document.createDocumentFragment();
  let charIndex = 0;

  words.forEach((word, wordIndex) => {
    const wordSpan = document.createElement("span");
    wordSpan.className = "quote-word";

    [...word].forEach((char, letterIndex) => {
      const charSpan = document.createElement("span");
      const drift = (letterIndex % 2) * 18;
      const wordPause = wordIndex * 120;
      const letterDelay = charIndex * 34 + drift + wordPause;

      charSpan.className = "quote-char";
      charSpan.textContent = char;
      charSpan.style.setProperty("--char-delay", `${letterDelay}ms`);
      wordSpan.appendChild(charSpan);
      charIndex += 1;
    });

    fragment.appendChild(wordSpan);

    if (wordIndex < words.length - 1) {
      const space = document.createElement("span");
      space.className = "quote-space";
      space.textContent = " ";
      fragment.appendChild(space);
    }
  });

  quoteEl.replaceChildren(fragment);
}

function revealHome() {
  window.setTimeout(() => {
    intro.classList.add("fade-out");
    home.classList.remove("hidden");

    window.requestAnimationFrame(() => {
      home.classList.add("revealed");
      home.setAttribute("aria-hidden", "false");
    });
  }, 1550);
}

window.addEventListener("load", () => {
  buildQuote();
  window.setTimeout(() => {
    intro.classList.add("quote-ready");
    revealHome();
  }, 220);
});
