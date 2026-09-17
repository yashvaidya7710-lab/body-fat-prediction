document.addEventListener("DOMContentLoaded", () => {

  // ---------------------------------------------------------------
  // Ambient particle field (canvas background)
  // ---------------------------------------------------------------
  const canvas = document.getElementById("fx-canvas");
  const ctx = canvas.getContext("2d");
  let particles = [];
  let W, H;

  function resizeCanvas() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  function initParticles() {
    const count = Math.min(90, Math.floor((W * H) / 18000));
    particles = Array.from({ length: count }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      r: Math.random() * 1.6 + 0.4,
      vy: -(Math.random() * 0.25 + 0.05),
      vx: (Math.random() - 0.5) * 0.15,
      hue: Math.random() > 0.5 ? "0,240,255" : "177,76,255",
      alpha: Math.random() * 0.5 + 0.15,
    }));
  }
  initParticles();
  window.addEventListener("resize", initParticles);

  function tickParticles() {
    ctx.clearRect(0, 0, W, H);
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.y < -10) { p.y = H + 10; p.x = Math.random() * W; }
      if (p.x < -10) p.x = W + 10;
      if (p.x > W + 10) p.x = -10;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.hue},${p.alpha})`;
      ctx.shadowColor = `rgba(${p.hue},0.9)`;
      ctx.shadowBlur = 6;
      ctx.fill();
    }
    requestAnimationFrame(tickParticles);
  }
  tickParticles();

  // ---------------------------------------------------------------
  // Slider <-> number field sync
  // ---------------------------------------------------------------
  const syncPairs = [
    "age", "weight_kg", "height_m", "max_bpm", "avg_bpm", "resting_bpm",
    "session_duration", "calories_burned", "water_intake", "workout_freq",
  ];
  syncPairs.forEach((id) => {
    const range = document.getElementById(id + "-r");
    const num = document.getElementById(id);
    if (!range || !num) return;
    range.addEventListener("input", () => { num.value = range.value; });
    num.addEventListener("input", () => {
      if (num.value !== "") range.value = num.value;
    });
  });

  // ---------------------------------------------------------------
  // Segmented control -> hidden select sync
  // ---------------------------------------------------------------
  function wireSegmented(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const targetSelect = document.getElementById(container.dataset.target);
    container.querySelectorAll(".seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        container.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        targetSelect.value = btn.dataset.value;
      });
    });
  }
  wireSegmented("gender-seg");
  wireSegmented("exp-seg");

  // ---------------------------------------------------------------
  // Prediction flow
  // ---------------------------------------------------------------
  const form = document.getElementById("predict-form");
  const submitBtn = document.getElementById("submit-btn");
  const errorNote = document.getElementById("form-error");
  const readout = document.getElementById("result-readout");
  const hint = document.getElementById("result-hint");
  const derivedBox = document.getElementById("result-derived");
  const derivedBmi = document.getElementById("derived-bmi");
  const derivedHrr = document.getElementById("derived-hrr");
  const ringFill = document.getElementById("ring-fill");
  const liquidFill = document.getElementById("liquid-fill");
  const liquidWave = document.getElementById("liquid-wave");
  const categoryBadge = document.getElementById("category-badge");
  const silhouetteWrap = document.getElementById("pulse-ring");

  const RING_CIRCUMFERENCE = 653.45; // 2 * PI * 104
  const SCALE_MAX = 45; // body fat % that maps to a fully-lit ring / full silhouette
  const LIQUID_TOP = 14;   // topmost y a full fill reaches
  const LIQUID_BASE = 200; // baseline y (empty)

  let currentDisplayed = 0;

  function categorize(value, gender) {
    const ranges = gender === "Male"
      ? [
          { max: 6, label: "Essential fat", cls: "cat-lean" },
          { max: 14, label: "Athletic", cls: "cat-lean" },
          { max: 18, label: "Fit", cls: "cat-fit" },
          { max: 25, label: "Average", cls: "cat-average" },
          { max: Infinity, label: "Above average", cls: "cat-above" },
        ]
      : [
          { max: 14, label: "Essential fat", cls: "cat-lean" },
          { max: 21, label: "Athletic", cls: "cat-lean" },
          { max: 25, label: "Fit", cls: "cat-fit" },
          { max: 32, label: "Average", cls: "cat-average" },
          { max: Infinity, label: "Above average", cls: "cat-above" },
        ];
    return ranges.find((r) => value < r.max);
  }

  function animateRing(targetValue) {
    const fraction = Math.max(0, Math.min(targetValue / SCALE_MAX, 1));
    ringFill.style.strokeDashoffset = RING_CIRCUMFERENCE * (1 - fraction);
  }

  function animateLiquid(targetValue) {
    const fraction = Math.max(0, Math.min(targetValue / SCALE_MAX, 1));
    const fillY = LIQUID_BASE - fraction * (LIQUID_BASE - LIQUID_TOP);
    liquidFill.setAttribute("y", fillY);
    liquidFill.setAttribute("height", LIQUID_BASE - fillY + 15);
    liquidWave.setAttribute("d",
      `M0 ${fillY} Q 12 ${fillY - 4} 25 ${fillY} T 50 ${fillY} T 75 ${fillY} T 100 ${fillY} V 210 H 0 Z`);
  }

  function animateCountUp(from, to, durationMs) {
    const start = performance.now();
    function step(now) {
      const elapsed = now - start;
      const t = Math.min(elapsed / durationMs, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      const value = from + (to - from) * eased;
      readout.textContent = value.toFixed(1);
      if (t < 1) {
        requestAnimationFrame(step);
      } else {
        readout.textContent = to.toFixed(1);
        currentDisplayed = to;
      }
    }
    requestAnimationFrame(step);
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorNote.textContent = "";
    submitBtn.disabled = true;
    submitBtn.querySelector(".btn-label").textContent = "SCANNING…";

    const genderValue = document.getElementById("gender").value;

    const payload = {
      age: document.getElementById("age").value,
      gender: genderValue,
      weight_kg: document.getElementById("weight_kg").value,
      height_m: document.getElementById("height_m").value,
      max_bpm: document.getElementById("max_bpm").value,
      avg_bpm: document.getElementById("avg_bpm").value,
      resting_bpm: document.getElementById("resting_bpm").value,
      session_duration: document.getElementById("session_duration").value,
      calories_burned: document.getElementById("calories_burned").value,
      workout_type: document.getElementById("workout_type").value,
      water_intake: document.getElementById("water_intake").value,
      workout_freq: document.getElementById("workout_freq").value,
      experience_level: document.getElementById("experience_level").value,
    };

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();

      if (!response.ok) {
        errorNote.textContent = data.error || "Something went wrong. Check the values and try again.";
        return;
      }

      const prediction = data.prediction;

      animateCountUp(currentDisplayed, prediction, 800);
      animateRing(prediction);
      animateLiquid(prediction);

      silhouetteWrap.classList.remove("pulsing");
      void silhouetteWrap.offsetWidth;
      silhouetteWrap.classList.add("pulsing");

      const cat = categorize(prediction, genderValue);
      categoryBadge.textContent = cat.label;
      categoryBadge.className = "category-badge " + cat.cls;
      categoryBadge.hidden = false;

      derivedBmi.textContent = data.bmi;
      derivedHrr.textContent = data.bpm_reserve;
      derivedBox.hidden = false;
      hint.textContent = "Estimate based on entered telemetry — not a medical measurement.";
    } catch (err) {
      errorNote.textContent = "Could not reach the server. Please try again.";
    } finally {
      submitBtn.disabled = false;
      submitBtn.querySelector(".btn-label").textContent = "RUN SCAN";
    }
  });
});
