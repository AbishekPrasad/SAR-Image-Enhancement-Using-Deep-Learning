let selectedImage = null;
let generatedContent = null;
let model = null;

// Load model when page loads
document.addEventListener('DOMContentLoaded', async () => {
    const loadingElement = document.getElementById('loadingModel');
    loadingElement.style.display = 'block';
    
    try {
        model = await tf.loadLayersModel('/static/js/tfjs-model/model.json');
        console.log("Model loaded successfully");
        loadingElement.style.display = 'none';
    } catch (error) {
        console.error("Error loading model:", error);
        loadingElement.style.display = 'none';
        alert("Failed to load the image processing model. Please refresh the page.");
    }
});

function previewImage() {
    const fileInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const originalSection = document.getElementById('originalSection');
    const enhancedSection = document.getElementById('enhancedSection');
    const downloadBtn = document.getElementById('downloadBtn');

    if (fileInput.files && fileInput.files[0]) {
        selectedImage = fileInput.files[0];
        const reader = new FileReader();

        reader.onload = function (event) {
            imagePreview.src = event.target.result;
            document.querySelector('.file-input-section').style.display = 'none';
            originalSection.style.display = 'block';
            enhancedSection.style.display = 'none';
            downloadBtn.style.display = 'none';
        };

        reader.readAsDataURL(selectedImage);
    }
}

async function generateOutput() {
    if (!selectedImage) {
        alert('Please upload an image first.');
        return;
    }

    if (!model) {
        alert('Model is still loading. Please wait a moment and try again.');
        return;
    }

    const outputImage = document.getElementById('outputImage');
    const downloadBtn = document.getElementById('downloadBtn');
    const enhancedSection = document.getElementById('enhancedSection');
    const loadingElement = document.getElementById('loadingModel');
    
    loadingElement.style.display = 'block';

    try {
        // Preprocess the image
        const tensor = await preprocessImage(selectedImage);
        
        // Generate prediction
        const prediction = model.predict(tensor);
        
        // Convert prediction to image
        const colorizedImage = await postprocessPrediction(prediction);
        
        // Display results
        outputImage.src = colorizedImage;
        generatedContent = colorizedImage;
        enhancedSection.style.display = 'block';
        downloadBtn.style.display = 'inline-block';
        
        // Clean up
        tf.dispose([tensor, prediction]);
    } catch (error) {
        console.error("Error during image processing:", error);
        alert("An error occurred while processing the image. Please try again.");
    } finally {
        loadingElement.style.display = 'none';
    }
}

async function preprocessImage(imageFile) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = async function(event) {
            const img = new Image();
            img.src = event.target.result;
            
            img.onload = () => {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = 256;
                canvas.height = 256;
                
                // Draw and resize image
                ctx.drawImage(img, 0, 0, 256, 256);
                
                // Convert to grayscale and normalize
                const imageData = ctx.getImageData(0, 0, 256, 256);
                const data = imageData.data;
                const grayscaleData = new Float32Array(256 * 256);
                
                for (let i = 0; i < data.length; i += 4) {
                    const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
                    grayscaleData[i/4] = (avg / 127.5) - 1; // Normalize to [-1, 1]
                }
                
                // Create tensor with shape [1, 256, 256, 1]
                const tensor = tf.tensor4d(grayscaleData, [1, 256, 256, 1]);
                resolve(tensor);
            };
        };
        reader.readAsDataURL(imageFile);
    });
}

async function postprocessPrediction(prediction) {
    return tf.tidy(() => {
        // Get AB channels from prediction
        const abChannels = prediction.squeeze();
        
        // Create dummy L channel (all zeros)
        const lChannel = tf.zerosLike(abChannels.slice([0, 0, 0], [256, 256, 1]));
        
        // Combine to LAB image
        const labImage = tf.concat([lChannel, abChannels], 2);
        
        // Convert LAB to RGB (simplified)
        const processed = labImage.mul(0.5).add(0.5).mul(255);
        const rgbImage = tf.clipByValue(processed, 0, 255);
        
        // Convert to image data URL
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 256;
        tf.browser.toPixels(rgbImage, canvas);
        
        return canvas.toDataURL();
    });
}

function downloadOutput() {
    if (generatedContent) {
        const link = document.createElement('a');
        link.href = generatedContent;
        link.download = 'enhanced_sar_image.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}