const CACHE_NAME = 'medai-v1';
const urlsToCache = [
    '/',
    '/static/images/logo1.png',
    '/static/css/styles.css',
    '/static/css/detect.css',
    '/templates/index.html',
    '/templates/detect.html',
    '/templates/clinics.html'
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(urlsToCache))
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => response || fetch(event.request))
    );
}); 