function attachCustomCopy() {
  document.querySelectorAll("button.md-clipboard").forEach((button) => {
    button.removeEventListener("click", handleCopy);
  });

  document.querySelectorAll("button.md-clipboard").forEach((button) => {
    button.addEventListener("click", handleCopy);
  });
}

function handleCopy(event) {
  event.preventDefault();
  const button = event.currentTarget;
  const codeBlock = document.querySelector(button.getAttribute('data-clipboard-target'));
  const codeBlockClone = codeBlock.cloneNode(true);
  codeBlockClone.querySelectorAll('.go').forEach(span => {
    const prev = span.previousSibling;
    if (prev && prev.nodeType === Node.TEXT_NODE) {
      prev.textContent = prev.textContent.replace(/[\r\n]+$/, '');
    }
  });
  codeBlockClone.querySelectorAll('.gp, .go').forEach(span => span.remove());
  navigator.clipboard.writeText(codeBlockClone.textContent || codeBlockClone.innerText);
}

document$.subscribe(() => {
  attachCustomCopy();
});
