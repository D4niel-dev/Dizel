(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const reduceMotion = window.matchMedia
    ? window.matchMedia("(prefers-reduced-motion: reduce)").matches
    : false;

  // ── Theme ──────────────────────────────────────────────────────────────
  const themeButton = $("[data-theme-toggle]");
  const themeMeta = $('meta[name="theme-color"]');
  const LIGHT = "#E8EAED";
  const DARK = "#131418";

  const storedTheme = () => {
    try {
      return localStorage.getItem("dizel.site.theme");
    } catch {
      return null;
    }
  };

  const applyTheme = (theme, persist) => {
    const dark = theme === "dark";
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
    if (themeMeta) themeMeta.setAttribute("content", dark ? DARK : LIGHT);
    if (themeButton) {
      themeButton.setAttribute("aria-pressed", String(dark));
      themeButton.setAttribute(
        "title",
        dark ? "Switch to light theme" : "Switch to dark theme",
      );
    }
    if (persist) {
      try {
        localStorage.setItem("dizel.site.theme", dark ? "dark" : "light");
      } catch {
        /* storage unavailable — the session still uses the chosen theme */
      }
    }
  };

  if (themeButton) {
    applyTheme(document.documentElement.dataset.theme || "light", false);
    themeButton.addEventListener("click", () => {
      const next =
        document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      applyTheme(next, true);
    });

    if (window.matchMedia) {
      const media = window.matchMedia("(prefers-color-scheme: dark)");
      const onChange = (event) => {
        if (storedTheme()) return;
        applyTheme(event.matches ? "dark" : "light", false);
      };
      if (media.addEventListener) media.addEventListener("change", onChange);
      else if (media.addListener) media.addListener(onChange);
    }
  }

  // ── Nav elevation ──────────────────────────────────────────────────────
  const navBar = $("[data-nav-bar]");
  if (navBar) {
    const syncNav = () => {
      navBar.toggleAttribute("data-scrolled", window.scrollY > 4);
    };
    window.addEventListener("scroll", syncNav, { passive: true });
    syncNav();
  }

  // ── Mobile menu ─────────────────────────────────────────────────────────
  const menuButton = $("#s-menu");
  const mobileMenu = $("#s-mobile-menu");
  if (menuButton && mobileMenu) {
    menuButton.addEventListener("click", () => {
      const open = mobileMenu.hidden;
      mobileMenu.hidden = !open;
      menuButton.setAttribute("aria-expanded", String(open));
    });
    const closeMenu = () => {
      mobileMenu.hidden = true;
      menuButton.setAttribute("aria-expanded", "false");
    };
    $$("a", mobileMenu).forEach((link) => {
      link.addEventListener("click", closeMenu);
    });
    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || mobileMenu.hidden) return;
      closeMenu();
      menuButton.focus();
    });
  }

  // ── Docs client tabs ───────────────────────────────────────────────────
  const selectTab = (tab) => {
    const name = tab.dataset.docTab;
    $$("[data-doc-tab]").forEach((item) => {
      const on = item === tab;
      item.setAttribute("aria-selected", String(on));
      item.tabIndex = on ? 0 : -1;
    });
    $$("[data-doc-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.docPanel !== name;
    });
  };
  $$("[data-doc-tab]").forEach((tab, index, all) => {
    tab.tabIndex = tab.getAttribute("aria-selected") === "true" ? 0 : -1;
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", (event) => {
      const step =
        event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
      let next = null;
      if (step) next = all[(index + step + all.length) % all.length];
      else if (event.key === "Home") next = all[0];
      else if (event.key === "End") next = all[all.length - 1];
      if (!next) return;
      event.preventDefault();
      selectTab(next);
      next.focus();
    });
  });

  // ── Copy buttons ───────────────────────────────────────────────────────
  const copyText = async (text) => {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      try {
        await navigator.clipboard.writeText(text);
        return;
      } catch {
        /* fall through to the selection fallback */
      }
    }
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.top = "-9999px";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const copied = document.execCommand("copy");
    area.remove();
    if (!copied) throw new Error("copy failed");
  };

  $$("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      const code = button.closest(".s-code")?.querySelector("code");
      if (!code) return;
      const label = button.textContent;
      const done = (text) => {
        button.textContent = text;
        button.classList.toggle("is-done", text === "Copied");
        window.setTimeout(() => {
          button.textContent = label;
          button.classList.remove("is-done");
        }, 1600);
      };
      try {
        await copyText(code.textContent);
        done("Copied");
      } catch {
        done("Copy failed");
      }
    });
  });

  // ── Router lab ─────────────────────────────────────────────────────────
  const routes = {
    "def add(": [0, 2],
    photosynthesis: [1, 3],
    "bầu cử": [0, 1],
    "2 + 2 =": [2, 3],
    "select *": [1, 2],
    Paris: [0, 3],
  };
  const tokenButtons = $$("[data-token]");
  const routeToken = $("[data-route-token]");
  const routeReadout = $("[data-route-readout]");
  const routeFrame = $(".s-route");
  const expertNodes = $$("[data-expert]");
  if (tokenButtons.length && routeToken && routeReadout) {
    let routeTimer;
    const selectToken = (value) => {
      tokenButtons.forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.token === value));
      });
      const active = routes[value] || [0, 2];
      routeToken.textContent = value;
      expertNodes.forEach((node, index) => {
        node.classList.toggle("is-active", active.includes(index));
      });
      routeReadout.textContent = `E${active[0]} + E${active[1]} · 2 of 4 awake`;
      if (routeFrame && !reduceMotion) {
        routeFrame.classList.remove("is-routing");
        void routeFrame.offsetWidth;
        routeFrame.classList.add("is-routing");
        window.clearTimeout(routeTimer);
        routeTimer = window.setTimeout(
          () => routeFrame.classList.remove("is-routing"),
          1100,
        );
      }
    };
    tokenButtons.forEach((button) => {
      button.addEventListener("click", () => selectToken(button.dataset.token));
    });
    // one authored entrance: the router fires once on arrival, then only on input
    if (routeFrame && !reduceMotion) {
      window.setTimeout(() => {
        routeFrame.classList.add("is-routing");
        window.setTimeout(() => routeFrame.classList.remove("is-routing"), 1200);
      }, 750);
    }
  }

  // ── Model card scrollspy ───────────────────────────────────────────────
  const paperSections = $$(".s-paper-main section[id]");
  const paperLinks = $$(".s-paper-side a");
  if (paperSections.length && paperLinks.length) {
    const syncPaper = () => {
      const y = window.scrollY + 120;
      let current = paperSections[0];
      for (const section of paperSections) {
        if (section.getBoundingClientRect().top + window.scrollY <= y) {
          current = section;
        }
      }
      paperLinks.forEach((link) => {
        link.classList.toggle(
          "is-active",
          link.getAttribute("href") === `#${current.id}`,
        );
      });
    };
    window.addEventListener("scroll", syncPaper, { passive: true });
    window.addEventListener("resize", syncPaper);
    syncPaper();
  }

  // ── Docs TOC scrollspy ─────────────────────────────────────────────────
  const tocLinks = $$('.s-docs-toc a[href^="#"]');
  const tocTargets = tocLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);
  if (tocLinks.length && tocTargets.length) {
    const syncToc = () => {
      const y = window.scrollY + 120;
      let current = tocTargets[0];
      for (const target of tocTargets) {
        if (target.getBoundingClientRect().top + window.scrollY <= y) {
          current = target;
        }
      }
      tocLinks.forEach((link) => {
        link.classList.toggle(
          "is-active",
          link.getAttribute("href") === `#${current.id}`,
        );
      });
    };
    window.addEventListener("scroll", syncToc, { passive: true });
    window.addEventListener("resize", syncToc);
    syncToc();
  }

  // ── Below-fold reveals ─────────────────────────────────────────────────
  if (!reduceMotion && "IntersectionObserver" in window) {
    const groups = [
      [".s-section-head", 0],
      [".s-matrix tbody tr", 45],
      [".s-links-list li", 45],
      [".s-evidence > *", 60],
      [".s-lab-head", 0],
      [".s-token-row", 0],
      [".s-route", 0],
      [".s-limit-list li", 40],
    ];
    const targets = [];
    groups.forEach(([selector, step]) => {
      $$(selector).forEach((element, index) => {
        if (element.getBoundingClientRect().top <= window.innerHeight * 0.92) return;
        element.classList.add("reveal");
        if (step) {
          element.style.setProperty(
            "--reveal-delay",
            `${Math.min(index, 5) * step}ms`,
          );
        }
        targets.push(element);
      });
    });
    if (targets.length) {
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            entry.target.classList.add("is-in");
            observer.unobserve(entry.target);
          });
        },
        { rootMargin: "0px 0px -6% 0px", threshold: 0.08 },
      );
      targets.forEach((element) => observer.observe(element));
    }
  }
})();
