function get_lyrics(){
  var your_seed = document.getElementById('seed').value;
  document.getElementById('lyrics').value = '';
  url = 'http://lotus.kuee.kyoto-u.ac.jp/~otani/hacku15/api/get_lyric.py';
  $.get(url, 
  { seed: your_seed},
  function(data){
    var content = data;
    document.getElementById('lyrics').value = content.replace(/<("[^"]*"|'[^']*'|[^'">])*>/g,'');
  });
};
