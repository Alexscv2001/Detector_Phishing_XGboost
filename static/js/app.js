const form = document.querySelector("#scanForm");
const input = document.querySelector("#urlInput");
const button = document.querySelector("#scanButton");
const panel = document.querySelector("#resultPanel");
const details = document.querySelector("#detailsSection");
const signals = document.querySelector("#signals");

function valueClass(value) {
  if (value > 0) return "good";
  if (value < 0) return "bad";
  return "neutral";
}

function renderResult(data) {
  panel.innerHTML = `
    <div class="verdict">
      <span class="badge ${data.status}">${data.label}</span>
      <span>${data.confidence}% confianza</span>
    </div>
    <p class="score">${data.risk_score}%</p>
    <p class="analyzed-url">${data.url}</p>
    <div class="metric-grid">
      <div class="metric">
        <span>Prob. phishing</span>
        <strong>${data.probabilities.phishing}%</strong>
      </div>
      <div class="metric">
        <span>Prob. legitimo</span>
        <strong>${data.probabilities.legitimate}%</strong>
      </div>
    </div>
  `;

  signals.innerHTML = data.explanations.map((item) => `
    <article class="signal">
      <span class="value ${valueClass(item.value)}">${item.value}</span>
      <strong>${item.name}</strong>
      <p>${item.detail}</p>
    </article>
  `).join("");
  details.hidden = false;
}

function renderError(message) {
  panel.innerHTML = `<p class="error">${message}</p>`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const url = input.value.trim();
  if (!url) return;

  button.disabled = true;
  button.textContent = "Analizando";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "No se pudo analizar la URL.");
    }
    renderResult(data);
  } catch (error) {
    renderError(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Analizar";
  }
});
