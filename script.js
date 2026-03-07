const quote = "Lose yourself to find yourself";
const intro = document.getElementById("intro");
const quoteEl = document.getElementById("quote");
const home = document.getElementById("home");

let index = 0;

function typeQuote() {
  if (index <= quote.length) {
    quoteEl.textContent = quote.slice(0, index);
    index += 1;
    setTimeout(typeQuote, 70);
    return;
  }

  setTimeout(() => {
    intro.classList.add("fade-out");
    home.classList.remove("hidden");
    home.classList.add("revealed");
    home.setAttribute("aria-hidden", "false");
  }, 1200);
}

window.addEventListener("load", () => {
  setTimeout(typeQuote, 400);
});