const status = document.getElementById('status');
const messages = document.getElementById('messages');
const messageInput = document.getElementById('messageInput');

const ctx = document.getElementById('myChart');
let labels = [];
let gay = [];
let lesbian = "#ec3750";

const myChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: labels,
        datasets: [{
            data: gay,
            borderWidth: 3,
            // fill: true,
            borderColor: lesbian,
            tension: 0.5,
            pointRadius: 0,
        }],
    },
    options: {
        animation: false,
        // animation:{
        //   duration: 120,
        //     easing: 'easeInOut'
        // },
        scales: {
            y: {
                beginAtZero: true
            }
        },
        plugins: {
            legend: {
                display: false
            }
        },
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
    messages.innerText = event.data;
    messages.scrollTop = messages.scrollHeight;
    if (labels.length < 40) labels.push("");
    gay.push(event.data);
    if (gay.length > 40){
        gay.shift();
    }
    const dataset = myChart.data.datasets[0];
    if(event.data === "0"){
        dataset.borderColor = "#8492a6";
        document.getElementById("honkshoo").innerText = "Dead"
        document.getElementById("honkshoo").style.color = "#8492a6"
    }else{
        dataset.borderColor = "#ec3750";
        document.getElementById("honkshoo").innerText = "Alive"
        document.getElementById("honkshoo").style.color = "#ec3750"
    }


    myChart.update('none');
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
