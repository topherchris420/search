(() => {
  const FILTER_KEYS = ["category", "source", "security_tier", "ontology_type"];
  const VIRTUALIZATION_THRESHOLD = 24;
  const VIRTUAL_ITEM_HEIGHT = 244;
  const VIRTUAL_OVERSCAN = 4;

  function emptyFacetState() {
    return {
      category: [],
      source: [],
      security_tier: [],
      ontology_type: [],
    };
  }

  const state = {
    query: "",
    filters: {
      category: "all",
      source: "all",
      security_tier: "all",
      ontology_type: "all",
    },
    filterUniverse: {
      category: [],
      source: [],
      security_tier: [],
      ontology_type: [],
    },
    facets: emptyFacetState(),
    facetPoolSize: 0,
    page: 1,
    pageSize: Number(window.APP_CONFIG?.defaultPageSize || 10),
    totalPages: 1,
    total: 0,
    results: [],
    loading: false,
    cacheHit: false,
    latencyMs: null,
    retries: 0,
    indexVersion: "--",
    requestSeq: 0,
    debounceTimer: null,
    activeController: null,
    virtualCleanup: null,
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
    statTotal: document.getElementById("stat-total"),
    statLatency: document.getElementById("stat-latency"),
    statCache: document.getElementById("stat-cache"),
    statRetry: document.getElementById("stat-retry"),
    statIndex: document.getElementById("stat-index"),
    categoryLabel: document.getElementById("category-label"),
    sourceLabel: document.getElementById("source-label"),
    securityLabel: document.getElementById("security-label"),
    ontologyLabel: document.getElementById("ontology-label"),
  };

  const THEME_KEY = "semantic_search_theme";
  const TOKEN_KEY = "semantic_search_token";

  function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
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

  function clearVirtualizedListeners() {
    if (typeof state.virtualCleanup === "function") {
      state.virtualCleanup();
    }
    state.virtualCleanup = null;
  }

  function setLoading(value) {
    state.loading = value;
    elements.loading.classList.toggle("hidden", !value);
    elements.results.classList.toggle("opacity-40", value);
    elements.prevPage.disabled = value || state.page <= 1;
    elements.nextPage.disabled = value || state.page >= state.totalPages;

    if (value) {
      clearVirtualizedListeners();
      const skeletonCount = Math.max(3, Math.min(state.pageSize, 8));
      elements.loading.innerHTML = Array.from({ length: skeletonCount })
        .map(
          () => `
          <div class="rounded-xl border border-white/10 bg-slate-900/55 p-4">
            <div class="mb-3 h-4 w-2/3 animate-pulse rounded bg-slate-700/70"></div>
            <div class="mb-2 h-3 w-full animate-pulse rounded bg-slate-800/80"></div>
            <div class="mb-2 h-3 w-11/12 animate-pulse rounded bg-slate-800/80"></div>
            <div class="h-2 w-1/2 animate-pulse rounded bg-cyan-700/60"></div>
          </div>
        `
        )
        .join("");
    }
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

        return { payload, attemptsUsed: attempt };
      } catch (error) {
        if (error.name === "AbortError") {
          throw error;
        }

        lastError = error;
        if (attempt >= attempts) {
          break;
        }
        await delay(160 * Math.pow(2, attempt - 1));
      }
    }

    throw lastError || new Error("Unknown API error");
  }

  function buildFilterOptions(values) {
    return ["all", ...(Array.isArray(values) ? values : [])];
  }

  function buildCountsMap(facetList) {
    const map = new Map();
    if (!Array.isArray(facetList)) {
      return map;
    }
    facetList.forEach((item) => {
      if (!item || typeof item !== "object") {
        return;
      }
      map.set(String(item.value), Number(item.count || 0));
    });
    return map;
  }

  function fillSelect(select, values, selected = "all", countsMap = new Map()) {
    const options = buildFilterOptions(values)
      .map((value) => {
        const countSuffix = value === "all" ? "" : ` (${countsMap.get(value) || 0})`;
        return `<option value="${escapeHtml(value)}">${escapeHtml(value === "all" ? "All" : value)}${countSuffix}</option>`;
      })
      .join("");

    select.innerHTML = options;
    const safeSelected = buildFilterOptions(values).includes(selected) ? selected : "all";
    select.value = safeSelected;
    return safeSelected;
  }

  function updateFilterLabels() {
    const facetHits = (entries) =>
      (Array.isArray(entries) ? entries : []).reduce((sum, item) => sum + Number(item.count || 0), 0);

    const categoryTypes = state.filterUniverse.category.length;
    const sourceTypes = state.filterUniverse.source.length;
    const securityTypes = state.filterUniverse.security_tier.length;
    const ontologyTypes = state.filterUniverse.ontology_type.length;

    const categoryHits = facetHits(state.facets.category);
    const sourceHits = facetHits(state.facets.source);
    const securityHits = facetHits(state.facets.security_tier);
    const ontologyHits = facetHits(state.facets.ontology_type);

    elements.categoryLabel.textContent = `Category (${categoryTypes} types, ${categoryHits} hits)`;
    elements.sourceLabel.textContent = `Source (${sourceTypes} types, ${sourceHits} hits)`;
    elements.securityLabel.textContent = `Security Tier (${securityTypes} types, ${securityHits} hits)`;
    elements.ontologyLabel.textContent = `Ontology Type (${ontologyTypes} types, ${ontologyHits} hits)`;
  }

  function refreshFilterSelects() {
    const categoryCounts = buildCountsMap(state.facets.category);
    const sourceCounts = buildCountsMap(state.facets.source);
    const securityCounts = buildCountsMap(state.facets.security_tier);
    const ontologyCounts = buildCountsMap(state.facets.ontology_type);

    state.filters.category = fillSelect(
      elements.category,
      state.filterUniverse.category,
      state.filters.category,
      categoryCounts
    );
    state.filters.source = fillSelect(
      elements.source,
      state.filterUniverse.source,
      state.filters.source,
      sourceCounts
    );
    state.filters.security_tier = fillSelect(
      elements.securityTier,
      state.filterUniverse.security_tier,
      state.filters.security_tier,
      securityCounts
    );
    state.filters.ontology_type = fillSelect(
      elements.ontologyType,
      state.filterUniverse.ontology_type,
      state.filters.ontology_type,
      ontologyCounts
    );

    updateFilterLabels();
  }

  function parseUrlState() {
    const params = new URLSearchParams(window.location.search);

    state.query = params.get("q") || "";

    const rawPage = Number(params.get("page") || "1");
    state.page = Number.isFinite(rawPage) && rawPage > 0 ? Math.floor(rawPage) : 1;

    const rawPageSize = Number(params.get("page_size") || String(window.APP_CONFIG?.defaultPageSize || 10));
    state.pageSize = Number.isFinite(rawPageSize) && rawPageSize > 0 ? Math.floor(rawPageSize) : Number(window.APP_CONFIG?.defaultPageSize || 10);

    FILTER_KEYS.forEach((key) => {
      const value = params.get(key);
      if (value) {
        state.filters[key] = value;
      }
    });

    elements.query.value = state.query;
    const pageSizeOptionExists = Array.from(elements.pageSize.options).some(
      (option) => Number(option.value) === state.pageSize
    );
    elements.pageSize.value = String(pageSizeOptionExists ? state.pageSize : Number(window.APP_CONFIG?.defaultPageSize || 10));
    state.pageSize = Number(elements.pageSize.value);
  }

  function syncUrlState() {
    const params = new URLSearchParams();

    if (state.query) {
      params.set("q", state.query);
    }
    if (state.page > 1) {
      params.set("page", String(state.page));
    }
    if (state.pageSize !== Number(window.APP_CONFIG?.defaultPageSize || 10)) {
      params.set("page_size", String(state.pageSize));
    }

    FILTER_KEYS.forEach((key) => {
      const value = state.filters[key];
      if (value && value !== "all") {
        params.set(key, value);
      }
    });

    const query = params.toString();
    const nextUrl = `${window.location.pathname}${query ? `?${query}` : ""}`;
    window.history.replaceState(null, "", nextUrl);
  }

  function securityTierStyle(tier) {
    const normalized = String(tier || "").toLowerCase();
    if (normalized === "secret") {
      return "bg-red-500/20 text-red-200 border border-red-400/40";
    }
    if (normalized === "confidential") {
      return "bg-amber-500/20 text-amber-200 border border-amber-400/40";
    }
    return "bg-emerald-500/20 text-emerald-200 border border-emerald-400/40";
  }

  function updateStats() {
    elements.statTotal.textContent = String(state.total);
    elements.statLatency.textContent = state.latencyMs == null ? "-- ms" : `${state.latencyMs} ms`;
    elements.statCache.textContent = state.cacheHit ? "HIT" : "MISS";
    elements.statRetry.textContent = String(state.retries);
    elements.statIndex.textContent = state.indexVersion || "--";
  }

  function buildResultCard(result, index, virtualized = false) {
    const scorePct = Math.max(2, Math.min(100, Math.round((Number(result.score || 0) + 1) * 50)));
    const tags = (result.tags || [])
      .map(
        (tag) =>
          `<span class="rounded-full border border-cyan-300/30 bg-cyan-500/10 px-2 py-0.5 text-xs text-cyan-100">${escapeHtml(tag)}</span>`
      )
      .join(" ");

    const delay = Math.min(index * 45, 360);
    const cardClass = virtualized
      ? "rounded-xl border border-white/12 bg-slate-900/60 p-4 shadow-lg shadow-black/20"
      : "result-card rounded-xl border border-white/12 bg-slate-900/60 p-4 shadow-lg shadow-black/20 transition hover:-translate-y-0.5 hover:border-cyan-300/40 hover:bg-slate-900/75";

    const style = virtualized
      ? `position:absolute; left:0; right:0; top:${index * VIRTUAL_ITEM_HEIGHT}px; height:${VIRTUAL_ITEM_HEIGHT - 12}px;`
      : `animation-delay:${delay}ms`;

    const snippet = escapeHtml(result.snippet || "");

    return `
      <article class="${cardClass}" style="${style}">
        <div class="mb-2 flex flex-wrap items-start justify-between gap-2">
          <h3 class="text-base font-bold text-slate-100">${escapeHtml(result.title)}</h3>
          <span class="rounded-full border border-cyan-300/40 bg-cyan-500/20 px-2 py-0.5 text-xs font-semibold text-cyan-100">Score ${Number(result.score || 0).toFixed(4)}</span>
        </div>
        <p class="mb-3 text-sm text-slate-300" style="display:-webkit-box; -webkit-box-orient:vertical; -webkit-line-clamp:${virtualized ? 3 : 4}; overflow:hidden;">${snippet}</p>
        <div class="mb-3 h-2 rounded bg-slate-800/90"><div class="h-2 rounded bg-gradient-to-r from-cyan-500 via-sky-400 to-emerald-400" style="width:${scorePct}%"></div></div>
        <div class="grid gap-2 text-xs text-slate-300 sm:grid-cols-2 lg:grid-cols-4">
          <span><strong>Category:</strong> ${escapeHtml(result.category)}</span>
          <span><strong>Source:</strong> ${escapeHtml(result.source)}</span>
          <span><strong>Ontology:</strong> ${escapeHtml(result.ontology_type)} / ${escapeHtml(result.ontology_id)}</span>
          <span class="inline-flex w-fit items-center rounded-full px-2 py-0.5 ${securityTierStyle(result.security_tier)}">${escapeHtml(result.security_tier)}</span>
        </div>
        <div class="mt-2 flex flex-wrap gap-2">${tags}</div>
      </article>
    `;
  }

  function renderVirtualizedResults() {
    const totalHeight = state.results.length * VIRTUAL_ITEM_HEIGHT;
    const viewportHeight = Math.min(Math.max(window.innerHeight * 0.62, 360), 760);

    elements.results.innerHTML = `
      <div id="virtual-scroll" class="overflow-y-auto pr-1" style="height:${Math.round(viewportHeight)}px;">
        <div id="virtual-spacer" style="position:relative; height:${totalHeight}px;"></div>
      </div>
    `;

    const scroller = document.getElementById("virtual-scroll");
    const spacer = document.getElementById("virtual-spacer");

    if (!scroller || !spacer) {
      return;
    }

    const renderWindow = () => {
      const scrollTop = scroller.scrollTop;
      const visibleCount = Math.ceil(scroller.clientHeight / VIRTUAL_ITEM_HEIGHT) + VIRTUAL_OVERSCAN * 2;
      const start = Math.max(0, Math.floor(scrollTop / VIRTUAL_ITEM_HEIGHT) - VIRTUAL_OVERSCAN);
      const end = Math.min(state.results.length, start + visibleCount);

      const windowHtml = [];
      for (let index = start; index < end; index += 1) {
        windowHtml.push(buildResultCard(state.results[index], index, true));
      }
      spacer.innerHTML = windowHtml.join("");
    };

    scroller.addEventListener("scroll", renderWindow, { passive: true });
    window.addEventListener("resize", renderWindow);

    state.virtualCleanup = () => {
      scroller.removeEventListener("scroll", renderWindow);
      window.removeEventListener("resize", renderWindow);
    };

    renderWindow();
  }

  function renderResults() {
    clearVirtualizedListeners();
    elements.results.innerHTML = "";

    if (!state.results.length) {
      elements.results.innerHTML =
        '<div class="rounded-xl border border-dashed border-white/15 bg-slate-950/30 p-5 text-sm text-slate-300">No results found for the current query and filters.</div>';
    } else if (state.results.length > VIRTUALIZATION_THRESHOLD) {
      renderVirtualizedResults();
    } else {
      const html = state.results.map((result, index) => buildResultCard(result, index, false)).join("");
      elements.results.innerHTML = html;
    }

    const virtualizationLabel =
      state.results.length > VIRTUALIZATION_THRESHOLD ? ` | virtualized ${state.results.length} items` : "";
    elements.pageIndicator.textContent = `Page ${state.page} of ${Math.max(1, state.totalPages)}`;
    elements.resultMeta.textContent = `${state.total} total results${virtualizationLabel}`;
    elements.cacheMeta.textContent = state.cacheHit
      ? `Cache: warm${state.facetPoolSize ? ` | facets top ${state.facetPoolSize}` : ""}`
      : `Cache: cold${state.facetPoolSize ? ` | facets top ${state.facetPoolSize}` : ""}`;
    elements.prevPage.disabled = state.loading || state.page <= 1;
    elements.nextPage.disabled = state.loading || state.page >= state.totalPages;
    updateStats();
  }

  async function loadFilters(signal = undefined) {
    const { payload } = await apiRequest("/api/filters", { method: "GET", signal });

    state.filterUniverse = {
      category: Array.isArray(payload.category) ? payload.category : [],
      source: Array.isArray(payload.source) ? payload.source : [],
      security_tier: Array.isArray(payload.security_tier) ? payload.security_tier : [],
      ontology_type: Array.isArray(payload.ontology_type) ? payload.ontology_type : [],
    };

    refreshFilterSelects();
  }

  function queueSearch({ resetPage = false, immediate = false } = {}) {
    if (resetPage) {
      state.page = 1;
    }

    if (state.debounceTimer) {
      clearTimeout(state.debounceTimer);
      state.debounceTimer = null;
    }

    if (immediate) {
      executeSearch();
      return;
    }

    state.debounceTimer = setTimeout(() => {
      executeSearch();
    }, Number(window.APP_CONFIG?.queryDebounceMs || 320));
  }

  async function executeSearch() {
    setError("");

    state.query = elements.query.value.trim();
    state.pageSize = Number(elements.pageSize.value || state.pageSize || 10);
    state.filters = {
      category: elements.category.value,
      source: elements.source.value,
      security_tier: elements.securityTier.value,
      ontology_type: elements.ontologyType.value,
    };

    syncUrlState();

    if (state.activeController) {
      state.activeController.abort();
    }

    const controller = new AbortController();
    state.activeController = controller;
    const requestId = ++state.requestSeq;
    const started = performance.now();

    setLoading(true);

    try {
      const { payload, attemptsUsed } = await apiRequest(
        "/api/search",
        {
          method: "POST",
          signal: controller.signal,
          body: JSON.stringify({
            query: state.query,
            filters: state.filters,
            page: state.page,
            page_size: state.pageSize,
          }),
        },
        3
      );

      if (requestId !== state.requestSeq) {
        return;
      }

      state.latencyMs = Math.round(performance.now() - started);
      state.retries = Math.max(0, attemptsUsed - 1);
      state.results = Array.isArray(payload.results) ? payload.results : [];
      state.totalPages = Math.max(1, Number(payload.total_pages || 1));
      state.total = Number(payload.total || 0);
      state.page = Number(payload.page || state.page);
      state.cacheHit = Boolean(payload.cache_hit);
      state.indexVersion = String(payload.index_version || "--").slice(0, 24);
      state.facets = payload.facets && typeof payload.facets === "object" ? payload.facets : emptyFacetState();
      state.facetPoolSize = Number(payload.facet_pool_size || 0);

      refreshFilterSelects();
      syncUrlState();
      renderResults();
    } catch (error) {
      if (error.name === "AbortError") {
        return;
      }
      if (requestId !== state.requestSeq) {
        return;
      }

      setError(error.message || "Search failed");
      state.results = [];
      state.total = 0;
      state.totalPages = 1;
      state.cacheHit = false;
      state.facets = emptyFacetState();
      state.facetPoolSize = 0;
      refreshFilterSelects();
      renderResults();
    } finally {
      if (requestId === state.requestSeq) {
        setLoading(false);
        state.activeController = null;
      }
    }
  }

  function resetForm() {
    state.page = 1;
    elements.query.value = "";
    FILTER_KEYS.forEach((key) => {
      state.filters[key] = "all";
    });
    elements.pageSize.value = String(window.APP_CONFIG?.defaultPageSize || 10);
  }

  function bindEvents() {
    elements.form.addEventListener("submit", (event) => {
      event.preventDefault();
      queueSearch({ resetPage: true, immediate: true });
    });

    elements.query.addEventListener("input", () => {
      queueSearch({ resetPage: true, immediate: false });
    });

    elements.resetBtn.addEventListener("click", () => {
      resetForm();
      refreshFilterSelects();
      queueSearch({ resetPage: true, immediate: true });
    });

    elements.refreshFilters.addEventListener("click", async () => {
      setError("");
      try {
        await loadFilters();
        queueSearch({ resetPage: false, immediate: true });
      } catch (error) {
        setError(error.message || "Failed to refresh filters");
      }
    });

    elements.prevPage.addEventListener("click", () => {
      if (state.page <= 1) {
        return;
      }
      state.page -= 1;
      queueSearch({ resetPage: false, immediate: true });
    });

    elements.nextPage.addEventListener("click", () => {
      if (state.page >= state.totalPages) {
        return;
      }
      state.page += 1;
      queueSearch({ resetPage: false, immediate: true });
    });

    elements.pageSize.addEventListener("change", () => {
      queueSearch({ resetPage: true, immediate: true });
    });

    [elements.category, elements.source, elements.securityTier, elements.ontologyType].forEach((select) => {
      select.addEventListener("change", () => {
        queueSearch({ resetPage: true, immediate: true });
      });
    });

    elements.themeToggle.addEventListener("click", () => {
      const isDark = document.documentElement.classList.contains("dark");
      updateTheme(isDark ? "light" : "dark");
    });

    elements.apiKey.addEventListener("input", async () => {
      const token = elements.apiKey.value.trim();
      if (token) {
        sessionStorage.setItem(TOKEN_KEY, token);
        if (window.APP_CONFIG?.zeroTrustEnabled) {
          try {
            await loadFilters();
            queueSearch({ resetPage: false, immediate: true });
          } catch (error) {
            setError(error.message || "Unable to authenticate with provided token");
          }
        }
      }
    });
  }

  async function bootstrap() {
    bootstrapTheme();

    const savedToken = sessionStorage.getItem(TOKEN_KEY);
    if (savedToken) {
      elements.apiKey.value = savedToken;
    }

    parseUrlState();
    bindEvents();

    if (window.APP_CONFIG?.zeroTrustEnabled && !elements.apiKey.value.trim()) {
      setError("Enter API key to load filters and start semantic search.");
      refreshFilterSelects();
      renderResults();
      return;
    }

    try {
      await loadFilters();
      queueSearch({ resetPage: false, immediate: true });
    } catch (error) {
      setError(error.message || "Initialization failed");
    }
  }

  bootstrap();
})();
