const status = document.getElementById('status');
const messages = document.getElementById('messages');
const messageInput = document.getElementById('messageInput');

const ctx = document.getElementById('myChart');
let labels = [];
let gay = [];

const myChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: labels,
        datasets: [{
            data: gay,
            borderWidth: 1
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});

// Connect to the WebSocket server
const ws = new WebSocket('ws://localhost:5555');

// Connection opened
ws.onopen = () => {
    status.textContent = 'Connected to server';
    status.style.color = 'green';
};

// Listen for messages
ws.onmessage = (event) => {
    const message = document.createElement('div');
    message.className = 'message';
    message.textContent = event.data;
    messages.appendChild(message);
    messages.scrollTop = messages.scrollHeight;
    labels.push("");
    gay.push(event.data);
    myChart.update();
};

// Handle errors
ws.onerror = (error) => {
    status.textContent = 'Error: ' + error.message;
    status.style.color = 'red';
};

// Handle connection close
ws.onclose = () => {
    status.textContent = 'Disconnected from server';
    status.style.color = 'red';
};

// Function to send a message
function sendMessage() {
    const message = messageInput.value.trim();
    if (message) {
        ws.send(message);
        messageInput.value = '';
    }
}

// Send message on Enter key
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});


