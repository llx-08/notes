/**
 * 首页列表无 <!-- more --> 时，按纯文本长度截断生成 excerpt，便于浏览多篇文章。
 * 某篇若已写 <!-- more --> 或 front-matter excerpt，则不受影响。
 * 长度在根 _config.yml 的 auto_excerpt_length 配置（默认 420）。
 */
hexo.extend.filter.register('after_post_render', data => {
    if (data.excerpt && String(data.excerpt).trim().length > 0) {
        return data;
    }
    if (!data.content || typeof data.content !== 'string') {
        return data;
    }
    const limit = Number(hexo.config.auto_excerpt_length) || 420;
    const plain = data.content.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
    if (plain.length <= limit) {
        return data;
    }
    const cut = plain.slice(0, limit).trimEnd() + '…';
    data.excerpt = `<p>${cut}</p>`;
    return data;
}, 20);
