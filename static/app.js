async function fetchJson(url) {
  const response = await fetch(url);
  const contentType = response.headers.get("content-type") || "";
  const rawText = await response.text();
  let payload = null;

  if (contentType.includes("application/json")) {
    try {
      payload = rawText ? JSON.parse(rawText) : null;
    } catch (error) {
      payload = null;
    }
  }

  if (!response.ok) {
    const message =
      payload?.detail?.message ||
      payload?.message ||
      rawText ||
      `Request failed with status ${response.status}`;
    throw new Error(message);
  }

  if (payload !== null) {
    return payload;
  }

  if (!rawText) {
    return {};
  }

  throw new Error(`Expected JSON response, received: ${rawText.slice(0, 120)}`);
}

function renderMetrics(metrics) {
  const container = document.getElementById("metrics-table");
  if (!metrics.length) {
    container.innerHTML = "<p>暂无指标，请先完成训练。</p>";
    return;
  }

  const headers = Object.keys(metrics[0]);
  const thead = headers.map((header) => `<th>${header}</th>`).join("");
  const tbody = metrics
    .map((row) => {
      return `<tr>${headers.map((header) => `<td>${row[header]}</td>`).join("")}</tr>`;
    })
    .join("");

  container.innerHTML = `<table><thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody></table>`;
}

function renderModels(models) {
  const container = document.getElementById("models-list");
  const recommendSelect = document.getElementById("recommend-model");
  const similarSelect = document.getElementById("similar-model");
  container.innerHTML = "";
  recommendSelect.innerHTML = "";
  similarSelect.innerHTML = "";

  models.forEach((model) => {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.dataset.status = model.status || "unknown";
    chip.innerHTML = `
      <span class="chip-dot"></span>
      <strong>${model.model}</strong>
      <span>${model.status}</span>
    `;
    container.appendChild(chip);

    if (model.status === "trained") {
      recommendSelect.insertAdjacentHTML("beforeend", `<option value="${model.model}">${model.model}</option>`);
    }
    if (model.status === "trained" && model.supports_item_similarity) {
      similarSelect.insertAdjacentHTML("beforeend", `<option value="${model.model}">${model.model}</option>`);
    }
  });

  if (!Array.from(recommendSelect.options).some((option) => option.value === "two_stage")) {
    recommendSelect.value = "svd";
  } else {
    recommendSelect.value = "two_stage";
  }
  if (Array.from(similarSelect.options).some((option) => option.value === "content")) {
    similarSelect.value = "content";
  }
}

function renderMovies(containerId, items) {
  const container = document.getElementById(containerId);
  if (!items.length) {
    container.innerHTML = "<p>暂无结果。</p>";
    return;
  }

  container.innerHTML = items
    .map(
      (item) => `
      <article class="movie-card">
        <h3>${item.title}</h3>
        <div class="movie-meta">
          <div>movieId: ${item.movieId}</div>
          <div>genres: ${item.genres || "-"}</div>
          <div>year: ${item.year ?? "-"}</div>
        </div>
        <div class="movie-extra">
          <div>director: ${item.director || "-"}</div>
          <div>actors: ${item.actors || "-"}</div>
          <div>genome: ${item.genome_tags || "-"}</div>
        </div>
        <span class="score">score ${Number(item.score).toFixed(4)}</span>
      </article>
    `
    )
    .join("");
}

function renderAbPlan(plan) {
  const container = document.getElementById("ab-plan");
  const variantCards = plan.variants
    .map(
      (variant) => `
      <article class="ab-card">
        <h3>${variant.name}</h3>
        <p>${variant.description}</p>
      </article>
    `
    )
    .join("");

  container.innerHTML = `
    <article class="ab-card">
      <h3>目标</h3>
      <p>${plan.goal}</p>
      <p><strong>周期建议：</strong>${plan.duration_hint}</p>
    </article>
    ${variantCards}
    <article class="ab-card">
      <h3>核心指标</h3>
      <p>${plan.north_star_metrics.join(" / ")}</p>
      <p><strong>护栏指标：</strong>${plan.guardrail_metrics.join(" / ")}</p>
      <p><strong>分层：</strong>${plan.segmentation.join(" / ")}</p>
    </article>
  `;
}

async function loadDashboard() {
  const [metricsPayload, modelsPayload, abPlan] = await Promise.all([
    fetchJson("/metrics/latest"),
    fetchJson("/models"),
    fetchJson("/ab-test/plan"),
  ]);

  renderMetrics(metricsPayload.metrics || []);
  renderModels(modelsPayload.models || []);
  renderAbPlan(abPlan);
}

document.getElementById("refresh-metrics").addEventListener("click", loadDashboard);

document.getElementById("recommend-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const userId = document.getElementById("user-id").value;
  const model = document.getElementById("recommend-model").value;
  const topK = document.getElementById("recommend-topk").value;
  try {
    const payload = await fetchJson(`/users/${userId}/recommendations?model=${model}&top_k=${topK}`);
    renderMovies("recommend-results", payload.results || []);
  } catch (error) {
    document.getElementById("recommend-results").innerHTML = `<p>${error.message}</p>`;
  }
});

document.getElementById("similar-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const movieId = document.getElementById("movie-id").value;
  const model = document.getElementById("similar-model").value;
  const topK = document.getElementById("similar-topk").value;
  try {
    const payload = await fetchJson(`/items/${movieId}/similar?model=${model}&top_k=${topK}`);
    renderMovies("similar-results", payload.results || []);
  } catch (error) {
    document.getElementById("similar-results").innerHTML = `<p>${error.message}</p>`;
  }
});

loadDashboard().catch((error) => {
  document.getElementById("metrics-table").innerHTML = `<p>${error.message}</p>`;
});
