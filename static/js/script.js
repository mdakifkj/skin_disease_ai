/**
 * DermaVision AI - Modern Healthcare Dashboard Interactions
 * Features: Dark Mode Toggle, Drag-and-Drop Upload, Client Validation, Chart.js Visualizations, History Search & Filter
 */

document.addEventListener("DOMContentLoaded", () => {
    initThemeToggle();
    initUploadPipeline();
    initHistoryTableFilters();
    initResultCharts();
});

/* ==========================================================================
   1. Dark Mode / Light Mode Toggle
   ========================================================================== */
function initThemeToggle() {
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const themeIcon = document.getElementById("themeIcon");
    const htmlEl = document.documentElement;

    const savedTheme = localStorage.getItem("dermavision-theme") || 
        (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

    setTheme(savedTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", () => {
            const currentTheme = htmlEl.getAttribute("data-bs-theme");
            const newTheme = currentTheme === "dark" ? "light" : "dark";
            setTheme(newTheme);
        });
    }

    function setTheme(theme) {
        htmlEl.setAttribute("data-bs-theme", theme);
        localStorage.setItem("dermavision-theme", theme);
        if (themeIcon) {
            if (theme === "dark") {
                themeIcon.className = "bi bi-sun-fill text-warning";
            } else {
                themeIcon.className = "bi bi-moon-stars-fill text-secondary";
            }
        }
    }
}

/* ==========================================================================
   2. Drag & Drop Upload & Form Processing
   ========================================================================== */
function initUploadPipeline() {
    const imageInput = document.getElementById("imageInput");
    const dropZone = document.getElementById("dropZone");
    const dropZonePrompt = document.getElementById("dropZonePrompt");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const removeImageBtn = document.getElementById("removeImageBtn");
    const uploadForm = document.getElementById("uploadForm");
    const loadingOverlay = document.getElementById("loadingOverlay");
    const submitBtn = document.getElementById("submitBtn");
    const fileInfoText = document.getElementById("fileInfoText");

    if (!imageInput || !dropZone) return;

    // Drag-over highlights
    ["dragenter", "dragover"].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        });
    });

    // Drop handler
    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length > 0) {
            imageInput.files = dt.files;
            handleFileValidationAndPreview(dt.files[0]);
        }
    });

    // File input change handler
    imageInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileValidationAndPreview(e.target.files[0]);
        }
    });

    // Remove image handler
    if (removeImageBtn) {
        removeImageBtn.addEventListener("click", (e) => {
            e.preventDefault();
            imageInput.value = "";
            imagePreview.src = "#";
            previewContainer.classList.add("d-none");
            dropZonePrompt.classList.remove("d-none");
            if (submitBtn) submitBtn.disabled = true;
        });
    }

    function handleFileValidationAndPreview(file) {
        const allowedTypes = ["image/jpeg", "image/png", "image/webp", "image/jpg"];
        if (!allowedTypes.includes(file.type)) {
            alert("Unsupported format. Please select a valid JPG, PNG, or WEBP image.");
            imageInput.value = "";
            return;
        }

        const maxBytes = 16 * 1024 * 1024;
        if (file.size > maxBytes) {
            alert("File size exceeds maximum allowed limit (16 MB). Please choose a smaller image.");
            imageInput.value = "";
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            if (fileInfoText) {
                const sizeKb = (file.size / 1024).toFixed(1);
                fileInfoText.textContent = `${file.name} (${sizeKb} KB)`;
            }
            dropZonePrompt.classList.add("d-none");
            previewContainer.classList.remove("d-none");
            if (submitBtn) submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Submit state animation
    if (uploadForm) {
        uploadForm.addEventListener("submit", (e) => {
            if (!imageInput.files || imageInput.files.length === 0) {
                e.preventDefault();
                alert("Please select or drop an image first.");
                return;
            }
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status"></span>Analyzing...';
            }
            if (loadingOverlay) {
                loadingOverlay.classList.remove("d-none");
                animateProgressSteps();
            }
        });
    }

    function animateProgressSteps() {
        const step1 = document.getElementById("stepValidate");
        const step2 = document.getElementById("stepModel");
        const step3 = document.getElementById("stepGradcam");

        setTimeout(() => { if (step1) step1.className = "text-success fw-semibold"; }, 400);
        setTimeout(() => { if (step2) step2.className = "text-success fw-semibold"; }, 1200);
        setTimeout(() => { if (step3) step3.className = "text-success fw-semibold"; }, 2000);
    }
}

/* ==========================================================================
   3. Result Page: Interactive Chart.js Visualizations
   ========================================================================== */
function initResultCharts() {
    const diseaseChartCanvas = document.getElementById("diseaseProbChart");
    if (!diseaseChartCanvas || typeof Chart === "undefined") return;

    const probabilitiesDataRaw = diseaseChartCanvas.getAttribute("data-chart-data");
    if (!probabilitiesDataRaw) return;

    try {
        const parsedData = JSON.parse(probabilitiesDataRaw);
        if (!Array.isArray(parsedData) || parsedData.length === 0) return;

        const labels = parsedData.map(item => item.display_name);
        const values = parsedData.map(item => item.confidence_percent || (item.probability * 100).toFixed(2));

        const isDark = document.documentElement.getAttribute("data-bs-theme") === "dark";
        const gridColor = isDark ? "rgba(255, 255, 255, 0.1)" : "rgba(0, 0, 0, 0.06)";
        const textColor = isDark ? "#cbd5e1" : "#475569";

        new Chart(diseaseChartCanvas, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Confidence Probability (%)",
                    data: values,
                    backgroundColor: [
                        "#4f46e5",
                        "#6366f1",
                        "#818cf8",
                        "#a5b4fc",
                        "#c7d2fe",
                        "#93c5fd",
                        "#60a5fa"
                    ],
                    borderRadius: 6,
                    borderSkipped: false,
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => ` Probability: ${ctx.raw}%`
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: gridColor },
                        ticks: {
                            color: textColor,
                            callback: (v) => `${v}%`
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: textColor, font: { weight: 500 } }
                    }
                }
            }
        });
    } catch (err) {
        console.warn("Chart rendering skipped:", err);
    }
}

/* ==========================================================================
   4. History Table: Search, Filter, and Sort Interactions
   ========================================================================== */
function initHistoryTableFilters() {
    const searchInput = document.getElementById("historySearchInput");
    const diseaseFilter = document.getElementById("diseaseFilterSelect");
    const historyTable = document.getElementById("historyTable");

    if (!historyTable || !searchInput) return;

    const rows = historyTable.querySelectorAll("tbody tr");

    function applyFilters() {
        const query = searchInput.value.toLowerCase().trim();
        const selectedDisease = diseaseFilter ? diseaseFilter.value.toLowerCase() : "";

        let visibleCount = 0;

        rows.forEach(row => {
            const filename = row.getAttribute("data-filename") || "";
            const disease = row.getAttribute("data-disease") || "";
            const rowText = row.textContent.toLowerCase();

            const matchesQuery = !query || rowText.includes(query) || filename.includes(query);
            const matchesDisease = !selectedDisease || disease === selectedDisease;

            if (matchesQuery && matchesDisease) {
                row.style.display = "";
                visibleCount++;
            } else {
                row.style.display = "none";
            }
        });

        const noResultsMsg = document.getElementById("noFilterResults");
        if (noResultsMsg) {
            noResultsMsg.classList.toggle("d-none", visibleCount > 0);
        }
    }

    searchInput.addEventListener("input", applyFilters);
    if (diseaseFilter) {
        diseaseFilter.addEventListener("change", applyFilters);
    }
}
