document.addEventListener('DOMContentLoaded', function() {
	// Only handle static carousels (those that exist in the initial HTML)
	const staticCarousels = document.querySelectorAll('.carousel:not([data-dynamic])');
	
	staticCarousels.forEach(carousel => {
		const closeBtn = carousel.querySelector('.carousel-close');
		if (closeBtn) {
			closeBtn.addEventListener('click', () => {
				carousel.style.display = 'none';
			});
		}
	});

	const staticThumbnails = document.querySelectorAll('.thumbnail:not([data-dynamic])');
	staticThumbnails.forEach(thumb => {
		thumb.addEventListener('click', () => {
			const carouselId = thumb.getAttribute('data-id');
			const carousel = document.getElementById(carouselId);
			if (carousel) {
				carousel.style.display = 'block';
			}
		});
	});
});