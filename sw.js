const V = 'cd-v5';
const CORE = ['./', 'index.html', 'kfrg.js', 'three.min.js', 'manifest.webmanifest', 'icon.png'];
self.addEventListener('install', e => { e.waitUntil(caches.open(V).then(c => c.addAll(CORE)).then(() => self.skipWaiting())); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== V).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
const put = (req, r) => { if (r && r.ok) { const c = r.clone(); caches.open(V).then(ch => ch.put(req, c)); } return r; };
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  const u = new URL(e.request.url);
  // world tiles and the big static files never change under the same address: cache first, no re-download
  if (/\/world\/t_[^/]+\.(bin|json)$/.test(u.pathname) || /three\.min\.js$|kfrg\.js$|icon\.png$/.test(u.pathname)) {
    e.respondWith(caches.match(e.request).then(hit => hit || fetch(e.request).then(r => put(e.request, r))));
    return;
  }
  // map index, real airports and terrain: always try the network first so a new world build shows up at once
  if (/\/world\/(index\.json|airports\.js|terrain\.js)$/.test(u.pathname)) {
    e.respondWith(fetch(e.request).then(r => put(e.request, r)).catch(() => caches.match(e.request, { ignoreSearch: true })));
    return;
  }
  // everything else: show the saved copy instantly and refresh it in the background for next time
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then(hit => {
    const net = fetch(e.request).then(r => put(e.request, r));
    return hit || net;
  }));
});
