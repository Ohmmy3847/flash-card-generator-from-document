// static/script.js
document.addEventListener('DOMContentLoaded', function() {
    const flashcard = document.querySelector('.flashcard');
    if (flashcard) {
        flashcard.addEventListener('click', function() {
            this.classList.toggle('flipped');
        });
    }

    const chatForm = document.getElementById('chat-form');
    const chatResponse = document.getElementById('chat-response');
    
    if (chatForm) {
        chatForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const question = document.getElementById('question-input').value;
            
            // Show loading message
            chatResponse.innerHTML = '<p>Thinking...</p>';
            
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `question=${encodeURIComponent(question)}`
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    chatResponse.innerHTML = `<p class="error">${data.error}</p>`;
                } else {
                    chatResponse.innerHTML = `<p><strong>Answer:</strong> ${data.answer}</p>`;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                chatResponse.innerHTML = '<p class="error">An error occurred while fetching the answer.</p>';
            });

            // Clear the input field
            document.getElementById('question-input').value = '';
        });
    }
});