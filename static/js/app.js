let isMouseDown = false; 
const pixelArray = new Array(28 * 28).fill(0); // Initialize array with 0s

// Mouse down and up events
document.addEventListener('mousedown', () => isMouseDown = true);
document.addEventListener('mouseup', () => isMouseDown = false);

// Function to handle pixel hover
function hoverPixel(pixel) {
    if (isMouseDown) {
        const index = pixel.getAttribute('data-index'); // To Get pixel's index
        pixel.classList.add('active');
        pixelArray[index] = 1; // Set the Exact index in the Array to 1
    }
}

// clear the drawing areax  
function clearDrawingArea() {
    const pixels = document.querySelectorAll('.pixel');
    pixels.forEach(pixel => {
        pixel.classList.remove('active');
    });
    console.log(pixelArray); // Neeraj's Magical Array
}