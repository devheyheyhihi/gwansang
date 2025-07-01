document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('imageInput');
    const uploadArea = document.getElementById('uploadArea');
    const loadingDiv = document.getElementById('loading');

    // 파일 선택 시 처리
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            handleFileUpload(file);
        }
    });

    // 드래그 앤 드롭 기능
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            imageInput.files = files;
            handleFileUpload(files[0]);
        }
    });

    // 업로드 영역 클릭 시 파일 선택 창 열기
    uploadArea.addEventListener('click', function(e) {
        // 버튼 자체를 클릭한 경우에는 기본 동작에 맡기고, 중복으로 click()을 호출하지 않도록 합니다.
        if (e.target.tagName === 'LABEL' || e.target.id === 'imageInput') {
            return;
        }
        imageInput.click();
    });

    function handleFileUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('이미지 파일만 업로드 가능합니다.');
            return;
        }

        // 파일 크기 체크 (10MB 제한)
        if (file.size > 10 * 1024 * 1024) {
            alert('파일 크기가 너무 큽니다. 10MB 이하의 파일을 선택해주세요.');
            return;
        }

        // 미리보기 표시
        const reader = new FileReader();
        reader.onload = function(e) {
            showPreview(e.target.result);
        };
        reader.readAsDataURL(file);

        // 자동으로 분석 시작
        setTimeout(() => {
            analyzeImage(file);
        }, 1000);
    }

    function showPreview(src) {
        uploadArea.innerHTML = `
            <div class="upload-content">
                <img src="${src}" alt="미리보기" style="max-width: 300px; max-height: 300px; border-radius: 10px; margin-bottom: 20px;">
                <p style="color: #8B4513; font-weight: bold;">이미지를 분석 중입니다...</p>
            </div>
        `;
    }

    function analyzeImage(file) {
        const formData = new FormData();
        formData.append('file', file);

        // 로딩 표시
        if (loadingDiv) {
            loadingDiv.style.display = 'block';
        }

        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 결과 페이지로 리다이렉트
                window.location.href = '/result?data=' + encodeURIComponent(JSON.stringify(data));
            } else {
                alert('분석 중 오류가 발생했습니다: ' + data.error);
                resetUploadArea();
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('분석 중 오류가 발생했습니다.');
            resetUploadArea();
        })
        .finally(() => {
            if (loadingDiv) {
                loadingDiv.style.display = 'none';
            }
        });
    }

    function resetUploadArea() {
        uploadArea.innerHTML = `
            <div class="upload-content">
                <div class="upload-icon">📸</div>
                <h3>얼굴 사진을 업로드하세요</h3>
                <p>정면을 바라보는 선명한 사진을 올려주시면<br>
                전통 관상학에 따른 운명 해석을 제공해드립니다</p>
                <label for="imageInput" class="upload-button">
                    사진 선택하기
                </label>
            </div>
        `;
    }
});
