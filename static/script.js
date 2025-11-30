document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");

    const sendMessage = async () => {
        const question = userInput.value.trim();
        if (!question) return;

        // Display user's message
        appendMessage(question, "user");
        userInput.value = "";
        userInput.style.height = 'auto'; // Reset height

        // Create a placeholder for the bot's response
        const botMessageElement = appendMessage("Thinking...", "bot", true);

        try {
            const response = await fetch("/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: question }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Handle the stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullResponse = "";
            botMessageElement.innerHTML = ""; // Clear "Thinking..."
            botMessageElement.classList.add("loading");

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    botMessageElement.classList.remove("loading");
                    break;
                }
                const chunk = decoder.decode(value, { stream: true });
                fullResponse += chunk;
                botMessageElement.innerHTML = marked.parse(fullResponse); // Use marked.js to render Markdown
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        } catch (error) {
            console.error("Error fetching response:", error);
            botMessageElement.innerHTML = "<p>Sorry, something went wrong. Please try again.</p>";
            botMessageElement.classList.remove("loading");
        }
    };

    const appendMessage = (text, sender, isLoading = false) => {
        const messageWrapper = document.createElement("div");
        messageWrapper.className = `message ${sender}-message`;
        if (isLoading) {
            messageWrapper.classList.add("loading");
        }
        
        const p = document.createElement("p");
        p.innerHTML = marked.parse(text); // Render markdown
        messageWrapper.appendChild(p);

        chatBox.appendChild(messageWrapper);
        chatBox.scrollTop = chatBox.scrollHeight;
        return p;
    };

    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });
    
    // Auto-resize textarea
    userInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
});