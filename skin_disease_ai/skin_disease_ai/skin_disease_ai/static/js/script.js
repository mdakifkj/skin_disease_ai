// DermaVision AI Client-side Script

document.addEventListener("DOMContentLoaded", () => {
    const imageInput = document.getElementById("imageInput");
    const dropZone = document.getElementById("dropZone");
    const dropZonePrompt = document.getElementById("dropZonePrompt");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const removeImageBtn = document.getElementById("removeImageBtn");
    const uploadForm = document.getElementById("uploadForm");
    const loadingOverlay = document.getElementById("loadingOverlay");
    const submitBtn = document.getElementById("submitBtn");

    if (imageInput && dropZone) {
        // Drag over & leave events
        ["dragenter", "dragover"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove("dragover");
            });
        });

        // Drop event
        dropZone.addEventListener("drop", (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                imageInput.files = files;
                handleFileSelect(files[0]);
            }
        });

        // Input change event
        imageInput.addEventListener("change", (e) => {
            const files = e.target.files;
            if (files && files.length > 0) {
                handleFileSelect(files[0]);
            }
        });

        // Remove image button
        if (removeImageBtn) {
            removeImageBtn.addEventListener("click", (e) => {
                e.preventDefault();
                imageInput.value = "";
                imagePreview.src = "#";
                previewContainer.classList.add("d-none");
                dropZonePrompt.classList.remove("d-none");
            });
        }
    }

    function handleFileSelect(file) {
        const validTypes = ["image/jpeg", "image/png", "image/webp", "image/jpg"];
        if (!validTypes.includes(file.type)) {
            alert("Please select a valid image file (JPG, PNG, or WEBP).");
            imageInput.value = "";
            return;
        }

        const maxMb = 16;
        if (file.size > maxMb * 1024 * 1024) {
            alert(`File size exceeds ${maxMb} MB. Please upload a smaller image.`);
            imageInput.value = "";
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZonePrompt.classList.add("d-none");
            previewContainer.classList.remove("d-none");
        };
        reader.readAsDataURL(file);
    }

    // Form submission animation
    if (uploadForm) {
        uploadForm.addEventListener("submit", () => {
            if (imageInput.files && imageInput.files.length > 0) {
                if (submitBtn) submitBtn.disabled = true;
                if (loadingOverlay) loadingOverlay.classList.remove("d-none");
            }
        });
    }
});
