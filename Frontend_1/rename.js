const fs = require('fs');
const files = [
  'src/App.js', 
  'src/index.js', 
  'src/api.js',
  'src/components/ChatArea.js',
  'src/components/Dashboard.js',
  'src/components/MessageInput.js',
  'src/components/Sidebar.js'
];
files.forEach(f => {
  if (fs.existsSync(f)) {
    fs.renameSync(f, f.replace('.js', '.jsx'));
    console.log('Renamed', f);
  }
});
