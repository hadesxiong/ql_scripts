// getPageSize.js
const QLApi = require('/ql/data/scripts/utils/base_ql/qlApi.js')

const ql = new QLApi('http://127.0.0.1:5700','bECuC0Exno-K','Nst_CnG7EhQh2GLbFIC2nMaq')

document.addEventListener('DOMContentLoaded', function() {
    var height = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);
    console.log('Page height:', height);
    // 设置环境变量
    ql.login().then(()=>{
        const env = {name: 'dailyFeeds_PageHeight',value:height};
        ql.updateEnv(env).then(success => {
            if (success) {console.log(`更新环境变量dailyFeeds_PageHeight成功,高度为${height}`)}
        })
    })
});