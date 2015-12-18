function get_lyrics(){
  document.getElementById('lyrics').value = '';
  url = 'http://lotus.kuee.kyoto-u.ac.jp/~otani/hacku15/api/get_lyric.py';
  $.get(url, function(data){
    var content = data.responseText;
    document.getElementById('lyrics').value = content.replace(/<("[^"]*"|'[^']*'|[^'">])*>/g,'');
  });
};