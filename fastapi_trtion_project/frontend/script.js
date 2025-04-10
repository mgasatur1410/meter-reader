document.getElementById('uploadBtn').addEventListener('click', async () => {
    const input = document.getElementById('imageInput');
    if (input.files.length === 0) {
        alert('Пожалуйста, выберите изображение.');
        return;
    }

    const file = input.files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/infer_image', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Ошибка сервера: ${response.status}`);
        }

        const data = await response.json();

        // Предполагаем, что data.result = [ [ text, conf ], [ text2, conf2 ], ... ]
        let output = '';
        if (data.result && Array.isArray(data.result)) {
            data.result.forEach(item => {
                // item[0] — распознанный текст, item[1] — уверенность
                if (Array.isArray(item) && item.length >= 1) {
                    output += item[0] + '\n';
                }
            });
        }

        if (!output) {
            output = 'Нет распознанных данных.';
        }

        document.getElementById('resultArea').textContent = output;
    } catch (err) {
        console.error(err);
        alert('Ошибка: ' + err.message);
    }
});
