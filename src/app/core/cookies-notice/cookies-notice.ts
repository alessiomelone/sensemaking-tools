module CookiesNoticeModule {
  function init(): void {
    loadScript();
    loadStyles();
  }

  function loadScript(): void {
    const script = document.createElement('script');
    script.src = 'https://www.gstatic.com/glue/cookienotificationbar/cookienotificationbar.min.js';
    script.type = 'text/javascript';
    script.setAttribute('data-glue-cookie-notification-bar-category', '2B');
    document.body.appendChild(script);
  }

  function loadStyles(): void {
    const link = document.createElement('link');
    link.href = 'https://www.gstatic.com/glue/cookienotificationbar/cookienotificationbar.min.css';
    link.rel = 'stylesheet';
    document.head.appendChild(link);
  }

  init();
}
