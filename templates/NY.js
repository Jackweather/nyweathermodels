let pngFiles = [];
let currentIndex = 0;
let isTemperatureView = false;
let preloadedImages = [];

// Save the current view to localStorage
function saveCurrentView(view) {
    localStorage.setItem("currentView", view);
}

// Restore the current view from localStorage
function restoreCurrentView() {
    const savedView = localStorage.getItem("currentView");
    return savedView ? savedView : "mslp"; // Default to "mslp" if no view is saved
}

// Update the fetchPngFiles function to use the restored view
async function fetchPngFiles(view = restoreCurrentView()) {
    saveCurrentView(view); // Save the selected view
    pngFiles = []; // Clear the previous PNG files
    currentIndex = 0; // Reset the current index
    preloadedImages = []; // Clear preloaded images
    const response = await fetch(`/list_pngs/${view}`);
    if (response.ok) {
        pngFiles = await response.json();
        if (pngFiles.length > 0) {
            document.getElementById("slider").max = pngFiles.length - 1;
            restoreSliderPosition(); // Restore slider position on page load
            updateImage();
            preloadAdjacentImages(); // Preload nearby images
        } else {
            document.getElementById("current-image").src = "";
            document.getElementById("slider").max = 0;
        }
    } else {
        console.error("Failed to fetch PNG files.");
    }
}

// Preload a limited number of images around the current index
function preloadAdjacentImages() {
    const range = 2; // Number of images to preload before and after the current index
    const start = Math.max(0, currentIndex - range);
    const end = Math.min(pngFiles.length - 1, currentIndex + range);

    for (let i = start; i <= end; i++) {
        if (!preloadedImages.includes(pngFiles[i])) {
            const img = new Image();
            img.src = `${pngFiles[i]}?t=${new Date().getTime()}`; // Cache-busting query parameter
            preloadedImages.push(pngFiles[i]);
        }
    }
}

// Update the displayed image based on the current index
function updateImage() {
    const imgElement = document.getElementById("current-image");
    if (pngFiles.length > 0) {
        const newSrc = `${pngFiles[currentIndex]}?t=${new Date().getTime()}`; // Cache-busting query parameter
        if (imgElement.src !== newSrc) { // Only update if the source is different
            imgElement.src = newSrc;
        }
        document.getElementById("slider").value = currentIndex;
        localStorage.setItem("sliderPosition", currentIndex); // Save slider position
    } else {
        imgElement.src = "";
    }
}

// Restore slider position from localStorage
function restoreSliderPosition() {
    const savedPosition = localStorage.getItem("sliderPosition");
    if (savedPosition !== null && pngFiles.length > 0) {
        currentIndex = Math.min(parseInt(savedPosition, 10), pngFiles.length - 1);
        document.getElementById("slider").value = currentIndex;
        updateImage();
    }
}

// Save slider position to localStorage on change
document.getElementById("slider").addEventListener("change", (event) => {
    localStorage.setItem("sliderPosition", event.target.value);
});

// Automatically fetch new PNGs at regular intervals
function startAutoFetch(interval = 5000) { // Updated interval: 5 seconds
    setInterval(() => {
        console.log("Fetching new PNGs...");
        fetchPngFiles();
    }, interval);
}

// Handle the slider change
document.getElementById("slider").addEventListener("input", (event) => {
    const newValue = parseInt(event.target.value, 10);
    if (Math.abs(newValue - currentIndex) === 1) { // Allow only single-step changes
        currentIndex = newValue;
        updateImage();
        preloadAdjacentImages(); // Preload adjacent images for smoother navigation
    } else {
        event.target.value = currentIndex; // Revert to the last valid value
    }
});

// Handle taps or clicks anywhere on the screen to update the slider
document.body.addEventListener("click", (event) => {
    const slider = document.getElementById("slider");
    const rect = slider.getBoundingClientRect();
    const clickX = event.clientX;

    // Check if the click is within the slider's horizontal bounds
    if (clickX >= rect.left && clickX <= rect.right) {
        const sliderWidth = rect.width;
        const relativeClickX = clickX - rect.left;
        const newSliderValue = Math.round((relativeClickX / sliderWidth) * slider.max);
        slider.value = newSliderValue;
        currentIndex = newSliderValue;
        updateImage();
    }
});

// Handle the Temperature button click
document.getElementById("temp-button").addEventListener("click", () => {
    isTemperatureView = true;
    const view = "temp";
    saveCurrentView(view); // Save the selected view
    handleViewButtonClick(view);
});

// Handle the Snow 8-to-1 button click
document.getElementById("snow-8-to-1-button").addEventListener("click", () => {
    isTemperatureView = false;
    const view = "snow_8_to_1";
    saveCurrentView(view); // Save the selected view
    handleViewButtonClick(view);
});

// Handle the Snow 10-to-1 button click
document.getElementById("snow-10-to-1-button").addEventListener("click", () => {
    isTemperatureView = false;
    const view = "snow_10_to_1";
    saveCurrentView(view); // Save the selected view
    handleViewButtonClick(view);
});

// Handle the Precip Type Rate button click
document.getElementById("precip-type-rate-button").addEventListener("click", () => {
    const view = "precip_type_rate";
    saveCurrentView(view);
    handleViewButtonClick(view);
});

// Handle the Wind button click
document.getElementById("wind-button").addEventListener("click", () => {
    const view = "wind";
    saveCurrentView(view);
    handleViewButtonClick(view);
});

// Handle the Total Precip button click
document.getElementById("total-precip-button").addEventListener("click", () => {
    const view = "total_precip";
    saveCurrentView(view);
    handleViewButtonClick(view);
});

// Add a delay before enabling navigation after switching views
async function handleViewButtonClick(view) {
    saveCurrentView(view);
    document.getElementById("slider").disabled = true;
    const buttons = document.querySelectorAll(".dropdown-content button");
    buttons.forEach(button => button.disabled = true);
    showLoadingOverlay();
    await fetchPngFiles(view);
    await preloadAllImages();
    hideLoadingOverlay();
    document.getElementById("slider").disabled = false;
    buttons.forEach(button => button.disabled = false);
}
function showLoadingOverlay() {
    document.getElementById("loading-overlay").style.display = "flex";
}
function hideLoadingOverlay() {
    document.getElementById("loading-overlay").style.display = "none";
}

async function preloadAllImages() {
    if (pngFiles.length === 0) return;
    let loadedCount = 0;
    await Promise.all(pngFiles.map(src => {
        return new Promise(resolve => {
            const img = new Image();
            img.onload = () => {
                loadedCount++;
                resolve();
            };
            img.onerror = () => resolve();
            img.src = `${src}?t=${new Date().getTime()}`;
        });
    }));
}

// Refresh the PNG list and update the slider and image
async function refreshPngList(view = restoreCurrentView()) {
    saveCurrentView(view); // Save the selected view
    try {
        const response = await fetch(`/list_pngs/${view}`);
        if (response.ok) {
            pngFiles = await response.json();
            currentIndex = 0; // Reset to the first image
            document.getElementById("slider").max = pngFiles.length - 1;
            document.getElementById("slider").value = currentIndex;
            updateImage();
            preloadAdjacentImages();
        } else {
            console.error("Failed to fetch PNG files.");
            pngFiles = [];
            currentIndex = 0;
            document.getElementById("slider").max = 0;
            updateImage();
        }
    } catch (error) {
        console.error("Error refreshing PNG list:", error);
    }
}

// Handle left and right arrow key presses for navigation
document.body.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft" && currentIndex > 0) {
        currentIndex -= 1; // Move to the previous image by 1
        document.getElementById("slider").value = currentIndex;
        updateImage();
        preloadAdjacentImages();
    } else if (event.key === "ArrowRight" && currentIndex < pngFiles.length - 1) {
        currentIndex += 1; // Move to the next image by 1
        document.getElementById("slider").value = currentIndex;
        updateImage();
        preloadAdjacentImages();
    }
});

// Ensure the slider increments by 1 on mouse drag
document.getElementById("slider").addEventListener("input", (event) => {
    const newValue = parseInt(event.target.value, 10);
    if (Math.abs(newValue - currentIndex) === 1) { // Allow only single-step changes
        currentIndex = newValue;
        updateImage();
        preloadAdjacentImages(); // Preload adjacent images for smoother navigation
    } else {
        event.target.value = currentIndex; // Revert to the last valid value
    }
});

// Toggle dropdown visibility
document.getElementById("dropdown-button").addEventListener("click", () => {
    const dropdown = document.getElementById("view-dropdown");
    dropdown.classList.toggle("active");
});

// Initialize the app with the restored view
refreshPngList();
startAutoFetch(); // Start auto-fetching PNGs
