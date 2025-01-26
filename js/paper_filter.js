document.addEventListener("DOMContentLoaded", () => {
    const buttons = document.querySelectorAll('button');
    const rows = document.querySelectorAll('#publications tr[data-tags]');

    buttons.forEach(button => {
      button.addEventListener('click', () => {
        const selectedTag = button.getAttribute('data-tag');
  
        rows.forEach(row => {
          const rowTags = row.getAttribute('data-tags').split(',');
  
          // Show or hide rows based on the selected tag
          if (selectedTag === 'all' || rowTags.includes(selectedTag)) {
            row.classList.remove('hidden');
          } else {
            row.classList.add('hidden');
          }
        });
        console.log(rows.classList); 
      });
    });
    
  });
  