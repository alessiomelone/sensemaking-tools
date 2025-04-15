declare global {
  interface Window {
    bundleError: boolean;
    dataLayer: any;
    handleBundleError: Function;
  }
}

/**
 * Execute JS that effects app layout above the fold on init
 */
import './header/header';
import './webfonts/webfonts';

/**
 * Then fetch the `core_defer` bundle, allowing the `core_init` bundle to be as small as possible
 */
setTimeout(() => {
  const script = document.createElement('script');
  script.type = 'text/javascript';
  script.src = './core_defer.bundle.js';
  script.onerror = () => window.handleBundleError();
  document.head.appendChild(script);
});
