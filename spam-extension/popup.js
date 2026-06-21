document.getElementById('scanBtn').addEventListener('click', async () => {
    const textInput = document.getElementById('emailText').value.trim();
    const resultBox = document.getElementById('result-box');
    
    if (!textInput) {
        alert("Please paste some text first!");
        return;
    }
    
    resultBox.style.display = "block";
    resultBox.className = ""; 
    resultBox.innerText = "Analyzing payload...";
    resultBox.style.color = "#555";

    try {
        // Send payload directly to your running local FastAPI backend
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                subject: "", // Optional optimization placeholder
                message: textInput
            })
        });

        if (!response.ok) {
            throw new Error(`API returned status code: ${response.status}`);
        }

        const data = await response.json();
        
        // Render prediction classification and associated confidence score
        if (data.status === "success") {
            const confidencePercentage = (data.confidence_score * 100).toFixed(1);
            resultBox.className = data.prediction; // Applies either .Spam or .Ham styling
            resultBox.innerText = `${data.prediction} (${confidencePercentage}% Confidence)`;
        } else {
            resultBox.innerText = "Error parsing response framework.";
        }

    } catch (error) {
        console.error("Inference fetch error:", error);
        resultBox.style.background = "#f9f9f9";
        resultBox.style.color = "#ff0000";
        resultBox.innerText = "Failed to communicate with API. Ensure FastAPI server is up.";
    }
});