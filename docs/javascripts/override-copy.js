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
  codeBlockClone.querySelectorAll('.gp, .go').forEach(span => span.remove());
  let rawText = codeBlockClone.textContent || codeBlockClone.innerText;

  let lines = rawText.split(/\r?\n/);
  while (lines.length > 0 && lines[lines.length - 1].trim() === '') {
    lines.pop();
  }
  navigator.clipboard.writeText(lines.join('\n'));
}

document$.subscribe(() => {
  attachCustomCopy();
});
