const V = 'cd-v3';
const CORE = ['./', 'index.html', 'kfrg.js', 'three.min.js', 'manifest.webmanifest', 'icon.png'];
self.addEventListener('install', e => { e.waitUntil(caches.open(V).then(c => c.addAll(CORE)).then(() => self.skipWaiting())); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== V).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
// cache first for instant launch; refresh in the background so the next launch has the newest version
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then(hit => {
    const net = fetch(e.request).then(r => { if (r && r.ok) { const c = r.clone(); caches.open(V).then(ch => ch.put(e.request, c)); } return r; });
    return hit || net;
  }));
});
