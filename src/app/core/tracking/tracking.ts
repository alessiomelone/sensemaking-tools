module TrackingModule {
  const gtmId = 'GTM-NB5LZM4M';

  function init(): void {
    window.dataLayer = window.dataLayer || [];
    window.dataLayer.push({'gtm.start': new Date().getTime(), event: 'gtm.js'});
    loadScript(gtmId);
  }

  function loadScript(gtmId: string): void {
    const script = document.createElement('script');
    script.async = true;
    script.src = `https://www.googletagmanager.com/gtm.js?id=${gtmId}`;
    document.head.appendChild(script);
  }

  init();
}
