// getPageSize.js
document.addEventListener('DOMContentLoaded', function() {
    var height = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);
    console.log('Page height:', height);
    // 这里你可以选择将高度保存到本地存储，或者通过其他方式传递给Python脚本
    localStorage.setItem('pageHeight', height);
});