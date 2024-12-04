import { predict } from './network.js';

document.addEventListener('DOMContentLoaded', function() {
    const inputCanvas = document.getElementById('input-screen');
    const resultCanvas = document.getElementById('result-screen');
    const submit = document.getElementById('submit-canvas');
    const inputCtx = inputCanvas.getContext('2d', { willReadFrequently: true });
    const resultCtx = resultCanvas.getContext('2d', { willReadFrequently: true });

    inputCtx.lineWidth = 35;
    inputCtx.lineCap = 'round';
    inputCtx.strokeStyle = 'white';

    const drawLabelsAndLine = () => {
        inputCtx.font = resultCtx.font = '30px Arial';
        inputCtx.fillStyle = resultCtx.fillStyle = 'white';
        inputCtx.fillText('Input', 20, 40);
        resultCtx.fillText('Output', 20, 40);
    
        // Draw white vertical line between the canvases
        resultCtx.beginPath();
        resultCtx.moveTo(resultCanvas.width, 0); // Start at the top middle
        resultCtx.lineTo(resultCanvas.width, resultCanvas.height); // Draw to the bottom middle
        resultCtx.lineWidth = 10; // Thickness of the line
        resultCtx.strokeStyle = 'white';
        resultCtx.stroke();
    };
    
    

    const addDrawingListeners = () => {
        let is_drawing = false;
        inputCanvas.addEventListener('mousedown', (e) => {
            is_drawing = true;
            inputCtx.beginPath();
            inputCtx.moveTo(e.offsetX, e.offsetY);
        });

        inputCanvas.addEventListener('mousemove', (e) => {
            if (is_drawing) {
                inputCtx.lineTo(e.offsetX, e.offsetY);
                inputCtx.stroke();
            }
        });

        ['mouseup', 'mouseleave'].forEach(type => 
            inputCanvas.addEventListener(type, () => {
                is_drawing = false;
                inputCtx.closePath();
            })
        );
    };

    const getBinaryArray = () => {
        const gridSize = 28, cellWidth = inputCanvas.width / gridSize, cellHeight = inputCanvas.height / gridSize;
        return Array.from({ length: gridSize * gridSize }, (_, i) => {
            const x = i % gridSize, y = Math.floor(i / gridSize);
            const { data: [r, g, b, alpha] } = inputCtx.getImageData(Math.floor((x + 0.5) * cellWidth), Math.floor((y + 0.5) * cellHeight), 1, 1);
            return (r > 128 && g > 128 && b > 128 && alpha === 255) ? 1 : 0;
        });
    };

    const displayPredictions = (probabilities) => {
        const sortedPredictions = probabilities.map((prob, index) => ({ index, prob }))
            .sort((a, b) => b.prob - a.prob);
    
        const startY = 180;
        resultCtx.font = '30px Arial';
        resultCtx.fillText('Prediction Table:', 20, startY - 10);
        resultCtx.fillText('Digit', 20, startY + 30);
        resultCtx.fillText('Probability', 200, startY + 30);
    
        sortedPredictions.slice(0, 5).forEach((item, index) => {
            resultCtx.fillText(`Digit ${item.index}`, 20, startY + 70 + index * 40); 
            resultCtx.fillText(`${Math.round(item.prob * 100)}%`, 200, startY + 70 + index * 40);
        });
    };
    

    drawLabelsAndLine();
    addDrawingListeners();

    submit.addEventListener('click', () => {
        predict(getBinaryArray()).then(displayPredictions);
    });
});
