const quote = "Know the end, value the journey.\nUnknown end, value the end.\nNo end in mind, value the now.";
const intro = document.getElementById("intro");
const quoteEl = document.getElementById("quote");
const home = document.getElementById("home");
const CHAR_ANIMATION_MS = 860;
const INTRO_START_DELAY_MS = 220;
const INTRO_HOLD_MS = 720;

function createWord(word, wordIndex, lineIndex, state) {
  const wordSpan = document.createElement("span");
  wordSpan.className = "quote-word";
  wordSpan.style.setProperty("--word-delay", `${wordIndex * 110}ms`);

  [...word].forEach((char, letterIndex) => {
    const charSpan = document.createElement("span");
    const drift = (letterIndex % 3) * 16;
    const wave = Math.sin((state.lineCharIndex + 1) * 0.9) * 10;
    const wordPause = wordIndex * 84;
    const lineOffset = lineIndex * 36;
    const letterDelay = Math.max(0, state.lineCharIndex * 24 + drift + wave + wordPause + lineOffset);

    charSpan.className = "quote-char";
    charSpan.textContent = char;
    charSpan.style.setProperty("--char-delay", `${letterDelay}ms`);
    wordSpan.appendChild(charSpan);
    state.maxDelay = Math.max(state.maxDelay, letterDelay);
    state.lineCharIndex += 1;
  });

  return wordSpan;
}

function buildQuote() {
  const lines = quote.split("\n");
  const fragment = document.createDocumentFragment();
  const state = { maxDelay: 0 };

  lines.forEach((line, lineIndex) => {
    const lineSpan = document.createElement("span");
    lineSpan.className = "quote-line";
    const words = line.trim().split(/\s+/).filter(Boolean);
    let wordIndex = 0;
    state.lineCharIndex = 0;

    words.forEach((word, indexInLine) => {
      lineSpan.appendChild(createWord(word, wordIndex, lineIndex, state));
      wordIndex += 1;

      if (indexInLine < words.length - 1) {
        const space = document.createElement("span");
        space.className = "quote-space";
        space.textContent = " ";
        lineSpan.appendChild(space);
      }
    });

    fragment.appendChild(lineSpan);

    if (lineIndex < lines.length - 1) {
      fragment.appendChild(document.createElement("br"));
    }
  });

  quoteEl.replaceChildren(fragment);
  return state.maxDelay + CHAR_ANIMATION_MS;
}

function revealHome(delay) {
  window.setTimeout(() => {
    intro.classList.add("fade-out");
    home.classList.remove("hidden");

    window.requestAnimationFrame(() => {
      home.classList.add("revealed");
      home.setAttribute("aria-hidden", "false");
    });
  }, delay);
}

// window.addEventListener("load", () => {
//   const quoteDuration = buildQuote();
//   window.setTimeout(() => {
//     intro.classList.add("quote-ready");
//     revealHome(quoteDuration + INTRO_HOLD_MS);
//   }, INTRO_START_DELAY_MS);
// });
