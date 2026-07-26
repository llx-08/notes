/**
 * Icarus does not expose a custom stylesheet setting. Load the site's own
 * stylesheet after the theme CSS so layout overrides survive `npm ci`.
 */
hexo.extend.filter.register('after_render:html', html => {
    if (typeof html !== 'string' || !html.includes('</head>')) {
        return html;
    }

    const root = String(hexo.config.root || '/').replace(/\/?$/, '/');
    const href = `${root}css/custom.css`;
    const marker = 'data-site-custom-styles';

    if (html.includes(marker)) {
        return html;
    }

    return html.replace(
        '</head>',
        `    <link ${marker} rel="stylesheet" href="${href}">\n</head>`
    );
});
