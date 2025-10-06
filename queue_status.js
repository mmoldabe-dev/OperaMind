/**
 * Автообновление статуса очереди и разговоров
 * Вставить в templates/base.html перед </body>
 */

// Обновление статуса очереди
function updateQueueStatus() {
    fetch('/api/queue/status')
        .then(response => response.json())
        .then(data => {
            const statusEl = document.getElementById('queue-status');
            if (statusEl) {
                statusEl.innerHTML = `
                    <div class="queue-info">
                        <span class="badge">🔄 В обработке: ${data.currently_processing}</span>
                        <span class="badge">📋 В очереди: ${data.queue_size}</span>
                        <span class="badge">✅ Обработано: ${data.total_processed}</span>
                        ${data.total_errors > 0 ? `<span class="badge error">❌ Ошибок: ${data.total_errors}</span>` : ''}
                    </div>
                `;
            }
        })
        .catch(err => console.error('Ошибка обновления статуса:', err));
}

// Обновление статуса конкретного разговора
function updateConversationStatus(conversationId) {
    fetch(`/api/conversation/${conversationId}/status`)
        .then(response => response.json())
        .then(data => {
            const statusEl = document.getElementById(`conv-status-${conversationId}`);
            if (statusEl) {
                statusEl.textContent = data.status_display;
                statusEl.className = `status-badge status-${data.status}`;
            }
            
            // Если завершено, перезагружаем страницу
            if (data.status === 'completed' && !data.has_analysis) {
                setTimeout(() => location.reload(), 1000);
            }
        })
        .catch(err => console.error('Ошибка обновления разговора:', err));
}

// Автообновление каждые 3 секунды
setInterval(updateQueueStatus, 3000);

// Обновляем статус разговора, если мы на его странице
const conversationId = document.body.dataset.conversationId;
if (conversationId) {
    setInterval(() => updateConversationStatus(conversationId), 3000);
}

// Первое обновление сразу
updateQueueStatus();
if (conversationId) {
    updateConversationStatus(conversationId);
}