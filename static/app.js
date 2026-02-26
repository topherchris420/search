(() => {
  const state = {
    query: "",
    filters: {
      category: "all",
      source: "all",
      security_tier: "all",
      ontology_type: "all",
    },
    page: 1,
    pageSize: Number(window.APP_CONFIG?.defaultPageSize || 10),
    totalPages: 1,
    total: 0,
    results: [],
    loading: false,
    cacheHit: false,
  };

  const elements = {
    form: document.getElementById("search-form"),
    query: document.getElementById("query"),
    category: document.getElementById("category"),
    source: document.getElementById("source"),
    securityTier: document.getElementById("security-tier"),
    ontologyType: document.getElementById("ontology-type"),
    pageSize: document.getElementById("page-size"),
    apiKey: document.getElementById("api-key"),
    resetBtn: document.getElementById("reset-btn"),
    refreshFilters: document.getElementById("refresh-filters"),
    prevPage: document.getElementById("prev-page"),
    nextPage: document.getElementById("next-page"),
    pageIndicator: document.getElementById("page-indicator"),
    resultMeta: document.getElementById("result-meta"),
    cacheMeta: document.getElementById("cache-meta"),
    loading: document.getElementById("loading"),
    results: document.getElementById("results"),
    errorBanner: document.getElementById("error-banner"),
    themeToggle: document.getElementById("theme-toggle"),
  };

  const THEME_KEY = "semantic_search_theme";
  const TOKEN_KEY = "semantic_search_token";

  function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function setError(message) {
    if (!message) {
      elements.errorBanner.classList.add("hidden");
      elements.errorBanner.textContent = "";
      return;
    }
    elements.errorBanner.textContent = message;
    elements.errorBanner.classList.remove("hidden");
  }

  function setLoading(value) {
    state.loading = value;
    elements.loading.classList.toggle("hidden", !value);
  }

  function updateTheme(mode) {
    const root = document.documentElement;
    const resolved = mode === "dark" ? "dark" : "light";
    root.classList.toggle("dark", resolved === "dark");
    localStorage.setItem(THEME_KEY, resolved);
  }

  function bootstrapTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved) {
      updateTheme(saved);
      return;
    }

    const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    updateTheme(prefersDark ? "dark" : "light");
  }

  async function apiRequest(path, options = {}, attempts = 3) {
    const token = elements.apiKey.value.trim();
    const headers = {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    };

    if (token) {
      headers.Authorization = `Bearer ${token}`;
      sessionStorage.setItem(TOKEN_KEY, token);
    }

    if (window.APP_CONFIG?.zeroTrustEnabled && !token) {
      throw new Error("API key is required because zero-trust mode is enabled.");
    }

    let lastError;
    for (let attempt = 1; attempt <= attempts; attempt += 1) {
      try {
        const response = await fetch(path, {
          ...options,
          headers,
        });

        const payload = await response.json().catch(() => ({}));

        if (!response.ok) {
          const message = payload.error || `Request failed with status ${response.status}`;
          throw new Error(message);
        }

        return payload;
      } catch (error) {
        lastError = error;
        if (attempt >= attempts) {
          break;
        }
        await delay(140 * Math.pow(2, attempt - 1));
      }
    }

    throw lastError || new Error("Unknown API error");
  }

  function buildFilterOptions(values) {
    return ["all", ...(Array.isArray(values) ? values : [])];
  }

  function fillSelect(select, values, selected = "all") {
    const options = buildFilterOptions(values)
      .map((value) => `<option value="${value}">${value === "all" ? "All" : value}</option>`)
      .join("");

    select.innerHTML = options;
    select.value = selected;
  }

  function renderResults() {
    elements.results.innerHTML = "";

    if (!state.results.length) {
      elements.results.innerHTML = '<div class="rounded-lg border border-dashed border-slate-300 p-5 text-sm text-slate-600 dark:border-slate-700 dark:text-slate-300">No results found.</div>';
    } else {
      const html = state.results
        .map((result) => {
          const scorePct = Math.max(0, Math.min(100, Math.round((result.score + 1) * 50)));
          const tags = (result.tags || []).map((tag) => `<span class="rounded-full bg-slate-200 px-2 py-0.5 text-xs dark:bg-slate-800">${tag}</span>`).join(" ");
          return `
            <article class="rounded-xl border border-slate-200 p-4 dark:border-slate-800">
              <div class="mb-2 flex flex-wrap items-start justify-between gap-2">
                <h3 class="text-base font-semibold">${result.title}</h3>
                <span class="rounded-full bg-brand-50 px-2 py-0.5 text-xs font-medium text-brand-700 dark:bg-brand-700/30 dark:text-brand-50">Score ${result.score.toFixed(4)}</span>
              </div>
              <p class="mb-3 text-sm text-slate-700 dark:text-slate-300">${result.snippet || ""}</p>
              <div class="mb-2 h-2 rounded bg-slate-200 dark:bg-slate-800"><div class="h-2 rounded bg-brand-500" style="width:${scorePct}%"></div></div>
              <div class="grid gap-2 text-xs text-slate-600 dark:text-slate-300 sm:grid-cols-2 lg:grid-cols-4">
                <span><strong>Category:</strong> ${result.category}</span>
                <span><strong>Source:</strong> ${result.source}</span>
                <span><strong>Tier:</strong> ${result.security_tier}</span>
                <span><strong>Ontology:</strong> ${result.ontology_type} / ${result.ontology_id}</span>
              </div>
              <div class="mt-2 flex flex-wrap gap-2">${tags}</div>
            </article>
          `;
        })
        .join("");
      elements.results.innerHTML = html;
    }

    elements.pageIndicator.textContent = `Page ${state.page} of ${Math.max(1, state.totalPages)}`;
    elements.resultMeta.textContent = `${state.total} total results`;
    elements.cacheMeta.textContent = state.cacheHit ? "Cache hit" : "Fresh query";
    elements.prevPage.disabled = state.page <= 1;
    elements.nextPage.disabled = state.page >= state.totalPages;
  }

  async function loadFilters() {
    const payload = await apiRequest("/api/filters", { method: "GET" });
    fillSelect(elements.category, payload.category, state.filters.category);
    fillSelect(elements.source, payload.source, state.filters.source);
    fillSelect(elements.securityTier, payload.security_tier, state.filters.security_tier);
    fillSelect(elements.ontologyType, payload.ontology_type, state.filters.ontology_type);
  }

  async function search() {
    setError("");
    setLoading(true);

    state.query = elements.query.value.trim();
    state.pageSize = Number(elements.pageSize.value || state.pageSize || 10);
    state.filters = {
      category: elements.category.value,
      source: elements.source.value,
      security_tier: elements.securityTier.value,
      ontology_type: elements.ontologyType.value,
    };

    try {
      const payload = await apiRequest("/api/search", {
        method: "POST",
        body: JSON.stringify({
          query: state.query,
          filters: state.filters,
          page: state.page,
          page_size: state.pageSize,
        }),
      });

      state.results = payload.results || [];
      state.totalPages = Math.max(1, Number(payload.total_pages || 1));
      state.total = Number(payload.total || 0);
      state.page = Number(payload.page || state.page);
      state.cacheHit = Boolean(payload.cache_hit);
      renderResults();
    } catch (error) {
      setError(error.message || "Search failed");
    } finally {
      setLoading(false);
    }
  }

  function resetForm() {
    state.page = 1;
    elements.query.value = "";
    [elements.category, elements.source, elements.securityTier, elements.ontologyType].forEach((select) => {
      select.value = "all";
    });
    elements.pageSize.value = String(window.APP_CONFIG?.defaultPageSize || 10);
  }

  function bindEvents() {
    elements.form.addEventListener("submit", async (event) => {
      event.preventDefault();
      state.page = 1;
      await search();
    });

    elements.resetBtn.addEventListener("click", async () => {
      resetForm();
      await search();
    });

    elements.refreshFilters.addEventListener("click", async () => {
      setError("");
      try {
        await loadFilters();
      } catch (error) {
        setError(error.message || "Failed to refresh filters");
      }
    });

    elements.prevPage.addEventListener("click", async () => {
      if (state.page <= 1) return;
      state.page -= 1;
      await search();
    });

    elements.nextPage.addEventListener("click", async () => {
      if (state.page >= state.totalPages) return;
      state.page += 1;
      await search();
    });

    elements.pageSize.addEventListener("change", async () => {
      state.page = 1;
      await search();
    });

    [elements.category, elements.source, elements.securityTier, elements.ontologyType].forEach((select) => {
      select.addEventListener("change", async () => {
        state.page = 1;
        await search();
      });
    });

    elements.themeToggle.addEventListener("click", () => {
      const isDark = document.documentElement.classList.contains("dark");
      updateTheme(isDark ? "light" : "dark");
    });

    elements.apiKey.addEventListener("input", () => {
      if (elements.apiKey.value.trim()) {
        sessionStorage.setItem(TOKEN_KEY, elements.apiKey.value.trim());
      }
    });
  }

  async function bootstrap() {
    bootstrapTheme();
    const savedToken = sessionStorage.getItem(TOKEN_KEY);
    if (savedToken) {
      elements.apiKey.value = savedToken;
    }

    bindEvents();

    try {
      await loadFilters();
      await search();
    } catch (error) {
      setError(error.message || "Initialization failed");
    }
  }

  bootstrap();
})();
