<!DOCTYPE HTML>
<html>

<head>
    <meta charset="utf-8">

    <title>predict.py (editing)</title>
    <link id="favicon" rel="shortcut icon" type="image/x-icon" href="/static/base/images/favicon-file.ico?v=f9f0a782d7d67b3a57bf7dce251d771b405c7890604576ec8b9a621a39d7670f6b43ffabef1e566f1cd741ee302e15977d9e1cf60bbacebafe75787b9916415c">
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <link rel="stylesheet" href="/static/components/jquery-ui/themes/smoothness/jquery-ui.min.css?v=fb45616eef2c454960f91fcd2a04efeda84cfacccf0c5d741ba2793dc1dbd6d3ab01aaae6485222945774c7d7a9a2e9fb87e0d8ef1ea96893aa6906147a371bb" type="text/css" />
    <link rel="stylesheet" href="/static/components/jquery-typeahead/dist/jquery.typeahead.min.css?v=b49500b9260947a0685fb13337f81a7c84bf7aa2d0ba7df8306472ffe38f229801e9092b00cc01d4fd07cd3cf3fb7f0c565cf7d2e822be66d2c309a62899171c" type="text/css" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    
<link rel="stylesheet" href="/static/components/codemirror/lib/codemirror.css?v=b6f0a3f439785c8cad12df4945b27df013b5a8bba2a563035fbe3629c9ff2fa2633207783cb7049650ee6512ae22d582eec2e47158d3f5ae3583c1400909e435">
<link rel="stylesheet" href="/static/components/codemirror/addon/dialog/dialog.css?v=e7f5db4afaccf8a00c10d62c3693642323d3fcf17604a4797803df04e4f144a944dc646c2fda20548df538ada775570127b2a15403996a171ced1769697f3ee4">

    <link rel="stylesheet" href="/static/style/style.min.css?v=e652140d1995d0b5418bf077dc177f9bba31f70a89deb2f6925ac49836be440045694026740e17d511027ab3d06dfbe82a1d9c29efcbb3cb8ad1d4b5d628c08c" type="text/css"/>
    

    <link rel="stylesheet" href="/custom/custom.css" type="text/css" />
    <script src="/static/components/es6-promise/promise.min.js?v=bea335d74136a63ae1b5130f5ac9a50c6256a5f435e6e09fef599491a84d834a8b0f011ca3eaaca3b4ab6a2da2d3e1191567a2f171e60da1d10e5b9d52f84184" type="text/javascript" charset="utf-8"></script>
    <script src="/static/components/react/react.production.min.js?v=9a0aaf84a316c8bedd6c2ff7d5b5e0a13f8f84ec02442346cba0b842c6c81a6bf6176e64f3675c2ebf357cb5bb048e0b527bd39377c95681d22468da3d5de735" type="text/javascript"></script>
    <script src="/static/components/react/react-dom.production.min.js" type="text/javascript"></script>
    <script src="/static/components/create-react-class/index.js?v=894ad57246e682b4cfbe7cd5e408dcd6b38d06af4de4f3425991e2676fdc2ef1732cbd19903104198878ae77de12a1996de3e7da3a467fb226bdda8f4618faec" type="text/javascript"></script>
    <script src="/static/components/requirejs/require.js?v=d37b48bb2137faa0ab98157e240c084dd5b1b5e74911723aa1d1f04c928c2a03dedf922d049e4815f7e5a369faa2e6b6a1000aae958b7953b5cc60411154f593" type="text/javascript" charset="utf-8"></script>
    <script>
      require.config({
          
          urlArgs: "v=20210703144829",
          
          baseUrl: '/static/',
          paths: {
            'auth/js/main': 'auth/js/main.min',
            custom : '/custom',
            nbextensions : '/nbextensions',
            kernelspecs : '/kernelspecs',
            underscore : 'components/underscore/underscore-min',
            backbone : 'components/backbone/backbone-min',
            jed: 'components/jed/jed',
            jquery: 'components/jquery/jquery.min',
            json: 'components/requirejs-plugins/src/json',
            text: 'components/requirejs-text/text',
            bootstrap: 'components/bootstrap/dist/js/bootstrap.min',
            bootstraptour: 'components/bootstrap-tour/build/js/bootstrap-tour.min',
            'jquery-ui': 'components/jquery-ui/jquery-ui.min',
            moment: 'components/moment/min/moment-with-locales',
            codemirror: 'components/codemirror',
            termjs: 'components/xterm.js/xterm',
            typeahead: 'components/jquery-typeahead/dist/jquery.typeahead.min',
          },
          map: { // for backward compatibility
              "*": {
                  "jqueryui": "jquery-ui",
              }
          },
          shim: {
            typeahead: {
              deps: ["jquery"],
              exports: "typeahead"
            },
            underscore: {
              exports: '_'
            },
            backbone: {
              deps: ["underscore", "jquery"],
              exports: "Backbone"
            },
            bootstrap: {
              deps: ["jquery"],
              exports: "bootstrap"
            },
            bootstraptour: {
              deps: ["bootstrap"],
              exports: "Tour"
            },
            "jquery-ui": {
              deps: ["jquery"],
              exports: "$"
            }
          },
          waitSeconds: 30,
      });

      require.config({
          map: {
              '*':{
                'contents': 'services/contents',
              }
          }
      });

      // error-catching custom.js shim.
      define("custom", function (require, exports, module) {
          try {
              var custom = require('custom/custom');
              console.debug('loaded custom.js');
              return custom;
          } catch (e) {
              console.error("error loading custom.js", e);
              return {};
          }
      })

    document.nbjs_translations = {"domain": "nbjs", "locale_data": {"nbjs": {"": {"domain": "nbjs"}}}};
    document.documentElement.lang = navigator.language.toLowerCase();
    </script>

    
    

</head>

<body class="edit_app "
 
data-base-url="/"
data-file-path="sagemaker-deployment/Project/serve/predict.py"

  
 

dir="ltr">

<noscript>
    <div id='noscript'>
      Jupyter Notebook requires JavaScript.<br>
      Please enable it to proceed. 
  </div>
</noscript>

<div id="header">
  <div id="header-container" class="container">
  <div id="ipython_notebook" class="nav navbar-brand"><a href="/tree" title='dashboard'>
      <img src='/static/base/images/logo.png?v=a2a176ee3cee251ffddf5fa21fe8e43727a9e5f87a06f9c91ad7b776d9e9d3d5e0159c16cc188a3965e00375fb4bc336c16067c688f5040c0c2d4bfdb852a9e4' alt='Jupyter Notebook'/>
  </a></div>

  

<span id="save_widget" class="pull-left save_widget">
    <span class="filename"></span>
    <span class="last_modified"></span>
</span>


  
  
  
  

    <span id="login_widget">
      
        <button id="logout" class="btn btn-sm navbar-btn">Logout</button>
      
    </span>

  

  
  
  </div>
  <div class="header-bar"></div>

  

<div id="menubar-container" class="container">
  <div id="menubar">
    <div id="menus" class="navbar navbar-default" role="navigation">
      <div class="container-fluid">
          <p  class="navbar-text indicator_area">
          <span id="current-mode" >current mode</span>
          </p>
        <button type="button" class="btn btn-default navbar-toggle" data-toggle="collapse" data-target=".navbar-collapse">
          <i class="fa fa-bars"></i>
          <span class="navbar-text">Menu</span>
        </button>
        <ul class="nav navbar-nav navbar-right">
          <li id="notification_area"></li>
        </ul>
        <div class="navbar-collapse collapse">
          <ul class="nav navbar-nav">
            <li class="dropdown"><a href="#" class="dropdown-toggle" data-toggle="dropdown">File</a>
              <ul id="file-menu" class="dropdown-menu">
                <li id="new-file"><a href="#">New</a></li>
                <li id="save-file"><a href="#">Save</a></li>
                <li id="rename-file"><a href="#">Rename</a></li>
                <li id="download-file"><a href="#">Download</a></li>
              </ul>
            </li>
            <li class="dropdown"><a href="#" class="dropdown-toggle" data-toggle="dropdown">Edit</a>
              <ul id="edit-menu" class="dropdown-menu">
                <li id="menu-find"><a href="#">Find</a></li>
                <li id="menu-replace"><a href="#">Find &amp; Replace</a></li>
                <li class="divider"></li>
                <li class="dropdown-header">Key Map</li>
                <li id="menu-keymap-default"><a href="#">Default<i class="fa"></i></a></li>
                <li id="menu-keymap-sublime"><a href="#">Sublime Text<i class="fa"></i></a></li>
                <li id="menu-keymap-vim"><a href="#">Vim<i class="fa"></i></a></li>
                <li id="menu-keymap-emacs"><a href="#">emacs<i class="fa"></i></a></li>
              </ul>
            </li>
            <li class="dropdown"><a href="#" class="dropdown-toggle" data-toggle="dropdown">View</a>
              <ul id="view-menu" class="dropdown-menu">
              <li id="toggle_header" title="Show/Hide the logo and notebook title (above menu bar)">
              <a href="#">Toggle Header</a></li>
              <li id="menu-line-numbers"><a href="#">Toggle Line Numbers</a></li>
              </ul>
            </li>
            <li class="dropdown"><a href="#" class="dropdown-toggle" data-toggle="dropdown">Language</a>
              <ul id="mode-menu" class="dropdown-menu">
              </ul>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="lower-header-bar"></div>


</div>

<div id="site">


<div id="texteditor-backdrop">
<div id="texteditor-container" class="container"></div>
</div>


</div>






    


<script src="/static/edit/js/main.min.js?v=4c84e8d3fee9ad3e10ef1f32d9c5dc76b5681e5de7600de6fd608c591f8b40260549172f36361f8ac8123a44adadd1fe142a2d56255a511c17f2d2f72122e918" type="text/javascript" charset="utf-8"></script>


<script type='text/javascript'>
  function _remove_token_from_url() {
    if (window.location.search.length <= 1) {
      return;
    }
    var search_parameters = window.location.search.slice(1).split('&');
    for (var i = 0; i < search_parameters.length; i++) {
      if (search_parameters[i].split('=')[0] === 'token') {
        // remote token from search parameters
        search_parameters.splice(i, 1);
        var new_search = '';
        if (search_parameters.length) {
          new_search = '?' + search_parameters.join('&');
        }
        var new_url = window.location.origin + 
                      window.location.pathname + 
                      new_search + 
                      window.location.hash;
        window.history.replaceState({}, "", new_url);
        return;
      }
    }
  }
  _remove_token_from_url();
</script>
</body>

</html>