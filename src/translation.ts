// src/translation.ts

interface Translations {
    [key: string]: {
      en: string;
      jp: string;
    };
  }
  const initialState = {
    language: 'jp'
  };
  
  export const translations: Translations = {
    'click_to_start_translation': {
      en: 'Click the screen to start translation',
      jp: '画面をクリックすると翻訳が始まります',
    },
    'click_to_start_recording': {
      en: 'Click the screen to start recording, and opening your mouth will stop',
      jp: '画面をクリックして録画が始まり、口を開けると終了します',
    },
    'enter_sign_name': {
      en: 'Enter the name of the sign language',
      jp: '手話の名前を入力してください',
    },
    'save': {
      en: 'Save',
      jp: '保存する',
    },
    'cancel': {
      en: 'Cancel',
      jp: 'キャンセル',
    },
    'no_saved_signs': {
      en: 'No saved signs.',
      jp: '保存された手話がありません。',
    },
    'search_placeholder': {
      en: 'Search...',
      jp: '検索...',
    },
    'logout': {
      en: 'Logout',
      jp: 'ログアウト',
    },
    'settings': {
      en: 'Settings',
      jp: '設定',
    },
    'dark_mode': {
      en: 'Dark Mode',
      jp: 'ダークモード',
    },
    'light_mode': {
      en: 'Light Mode',
      jp: 'ライトモード',
    },
    'language': {
      en: 'Language',
      jp: '言語',
    },
    'english': {
      en: 'English',
      jp: 'English',
    },
    'japanese': {
      en: '日本語',
      jp: '日本語',
    },
    'translate': {
      en: 'Translate',
      jp: '翻訳',
    },
    'record': {
      en: 'Record',
      jp: '録画',
    },
    'files': {
      en: 'Files',
      jp: 'ファイル',
    },
    'documentation': {
      en: 'Documentation',
      jp: 'ドキュメント',
    },
    'upload': {
      en: 'Upload',
      jp: 'アップロード',
    },
    'download': {
      en: 'Download',
      jp: 'ダウンロード',
    },
    'delete': {
      en: 'Delete',
      jp: '削除する',
    },
    'save_recording': {
      en: 'Save Recording',
      jp: '録画を保存',
    },
    'cancel_saving': {
      en: 'Cancel Saving',
      jp: '保存をキャンセル',
    },
    'preview': {
      en: 'Preview',
      jp: 'プレビュー',
    },
    'close': {
      en: 'Close',
      jp: '閉じる',
    },
    'login': {
      en: 'Login',
      jp: 'ログイン',
    },
    'signup': {
      en: 'Sign Up',
      jp: 'サインアップ',
    },
    'username': {
      en: 'Username',
      jp: 'ユーザー名',
    },
    'password': {
      en: 'Password',
      jp: 'パスワード',
    },
    'remember_me': {
      en: 'Remember Me',
      jp: 'ログイン状態を保持',
    },
    'no_account': {
      en: "Don't have an account?",
      jp: 'アカウントをお持ちでないですか？',
    },
    'have_account': {
      en: 'Already have an account?',
      jp: '既にアカウントをお持ちですか？',
    },
    'login_error': {
      en: 'Login failed. Please check your credentials.',
      jp: 'ログインに失敗しました。入力情報を確認してください。',
    },
    'signup_error': {
      en: 'Sign up failed. Please try again.',
      jp: 'サインアップに失敗しました。もう一度お試しください。',
    },
    'saved_files': {
    en: 'Saved Files',
    jp: '保存されたファイル',
    },
    'search_folders': {
        en: 'Search Folders',
        jp: 'フォルダ検索',
    },
    'delete_selected': {
        en: 'Delete Selected',
        jp: '選択したものを削除',
    },
    'add_description': {
        en: 'Click to add description...',
        jp: '説明を追加するにはクリック...',
    },
    'browser_not_support': {
        en: 'Your browser does not support the video tag.',
        jp: 'お使いのブラウザはビデオタグをサポートしていません。',
    },
    'help': {
      en: 'Help',
      jp: 'ヘルプ',
    },
    'Help': {
      en: 'Help',
      jp: 'ヘルプ',
    },
    'educational_resource': {
      en: 'Educational Resource',
      jp: '教育リソース',
    },
    'Want to Learn?': {
      en: 'Want to Learn?',
      jp: '手話を学ぶ',
    },
    'Close Help Modal': {
      en: 'Close Help Modal',
      jp: 'ヘルプを閉じる',
    },
    'Recording Instructions': {
      en: 'Recording Instructions',
      jp: '録画の手順',
    },
    'Starting the Recording': {
      en: 'Starting the Recording',
      jp: '録画の開始',
    },
    'Ending the Recording': {
      en: 'Ending the Recording',
      jp: '録画の終了',
    },
    'Saving Your Sign': {
      en: 'Saving Your Sign',
      jp: '手話の保存',
    },
    'Recording Tips': {
      en: 'Recording Tips',
      jp: '録画のヒント',
    },
    'new learning': {
      en: 'If you don’t know where to start, try starting from learning basic signs on Handspeak. Click the link below to visit the website. Once there, open any word, watch the video, and try recording it here!',
      jp: 'どこから始めればよいかわからない場合は、Handspeakで基本的なアメリカ手話を学んでみてください。以下のリンクをクリックしてウェブサイトにアクセスし、任意の単語を開き、動画を見て、Handyで録画してみてください！',
    },
    'hide_overlay': {
      en: 'Hide Overlay',
      jp: 'オーバーレイ（人型の枠）を隠す',
    },
    'show_overlay': {
      en: 'Show Overlay',
      jp: 'オーバーレイ（人型の枠）を表示',
    },
      'This is the Record feature.': {
        en: 'This is the Record feature. Here\'s how to use it:',
        jp: 'これは録画機能です。使い方は以下の通りです：',
      },
      'Press anywhere on the screen to start. The blur will disappear, and after a 3-second countdown, you can begin signing. Make sure to adjust your position so your face and hands align with the person outline on the screen.': {
        en: 'Press anywhere on the screen to start. The blur will disappear, and after a 3-second countdown, you can begin signing. Make sure to adjust your position so your face and hands align with the person outline on the screen.',
        jp: '画面のどこかを押して開始します。ぼかしが消え、3秒のカウントダウンの後に手話を始めることができます。顔と手が画面の人物の輪郭と一致するように位置を調整してください。',
      },
      'To end the recording, simply open your mouth. This will stop the recording automatically.': {
        en: 'To end the recording, simply open your mouth. This will stop the recording automatically.',
        jp: '録画を終了するには、口を開くだけです。これにより録画が自動的に停止します。',
      },
      'Once the recording ends, enter a name for your sign and press "Save" to store it. You can view or delete past recordings in the "Files" section.': {
        en: 'Once the recording ends, enter a name for your sign and press "Save" to store it. You can view or delete past recordings in the "Files" section.',
        jp: '録画が終了したら、手話の名前を入力し、「保存」を押して保存します。「ファイル」セクションで過去の録画を表示または削除できます。',
      },
        'keep_hands_in_frame': {
        en: 'Always keep both hands inside the frame throughout the recording.',
        jp: '録画中は常に両手をフレーム内に収めてください。',
        },
        'one_hand_sign_tip': {
        en: 'For one-handed signs, keep your left hand on your stomach, as shown in the person outline.',
        jp: '片手の手話の場合は、人物の輪郭に示されているように、左手をお腹に置いてください。',
        },
        'copy_subtitles': {
            en: 'Copy as text',
            jp: 'テキストとしてコピー',
          },
          'How to Use': {
            en: 'How to Use',
            jp: '使用方法',
          },
          'Help Instructions': {
            en: 'Help Instructions',
            jp: 'ヘルプの手順',
          },
          'Start Signing': {
            en: 'Start Signing',
            jp: '手話を開始',
          },
          'Positioning for Signs': {
            en: 'Positioning for Signs',
            jp: '手話の位置',
          },
          'Device Limitations': {
            en: 'Device Limitations',
            jp: 'デバイスの制限',
          },
          'Copying Subtitles': {
            en: 'Copying Subtitles',
            jp: '字幕のコピー',
          },
          'Disabling the Outline': {
            en: 'Disabling the Outline',
            jp: 'アウトラインを無効にする',
          },
          'Copied to your clipboard!': {
            en: 'Copied to your clipboard!',
            jp: 'クリップボードにコピーされました！',
          },
          'close_copy_popup': {
            en: 'Close Copy Popup',
            jp: 'コピーのポップアップを閉じる',
          },
          'Tap anywhere on the screen to begin. When the blur disappears, align your face with the outline for accurate sign recognition.': {
            en: 'Tap anywhere on the screen to begin. When the blur disappears, align your face with the outline for accurate sign recognition.',
            jp: '画面のどこかをタップして開始します。ぼかしが消えたら、顔を輪郭に合わせて正確な手話認識を行います。',
          },
          '- For two-handed signs, stay within the frame.': {
            en: 'For two-handed signs, stay within the frame.',
            jp: '両手を使う手話の場合、フレーム内に収めてください。',
          },
          '- For one-handed signs (like the alphabet), rest your left hand on your stomach, as shown in the outline.': {
            en: 'For one-handed signs (like the alphabet), rest your left hand on your stomach, as shown in the outline.',
            jp: '片手の手話（アルファベットなど）の場合、輪郭に示されているように左手をお腹に置いてください。',
          },
          'Tracking may be slower on some devices. If subtitles aren’t appearing, try signing a bit slower.': {
            en: 'Tracking may be slower on some devices. If subtitles aren’t appearing, try signing a bit slower.',
            jp: '一部のデバイスではトラッキングが遅くなる場合があります。字幕が表示されない場合は、少しゆっくり手話をしてみてください。',
          },
          'Press the "Copy" button under this help section to copy subtitles.': {
            en: 'Press the "Copy" button under this help section to copy subtitles.',
            jp: 'このヘルプセクションの下にある「コピー」ボタンを押して字幕をコピーします。',
          },
          'To hide the outline, tap the button in the bottom-right corner of the screen.': {
            en: 'To hide the outline, tap the button in the bottom-right corner of the screen.',
            jp: 'アウトラインを非表示にするには、画面の右下にあるボタンをタップしてください。',
          },
          'This app translates sign language into subtitles below your screen. Here’s how to get started:' : {
            en: 'This app translates sign language into subtitles below your screen. Here’s how to get started:',
            jp: 'このアプリは手話を画面下の字幕に翻訳します。始め方は以下の通りです：',
          },
          'Copy as text': {
            en: 'Copy as text',
            jp: 'テキストとしてコピーする',
          },
          "screen_paused": {
            "en": "Screen Paused",
            "jp": "画面が一時停止されています"
          },
          "press_help_for_instructions": {
            "en": "Press help for detailed instructions",
            "jp": "詳しい手順についてはヘルプを押してください"
          },
         "tap_to_start_recording": {
            "en": "Tap anywhere to start recording",
            "jp": "録画を開始するには、どこでもタップしてください"
          },
          "tap_to_start_translating": {
            "en": "Tap anywhere to start translation",
            "jp": "翻訳を開始するには、どこでもタップしてください"
          },
          "documentation_popup_title": {
            "en": "Documentation Access",
            "jp": "ドキュメントアクセス"
          },
          "documentation_popup_description": {
            "en": "This link will redirect you to the Handy website documentation. Please note that access might be restricted until the full public release.",
            "jp": "このリンクはHandyウェブサイトのドキュメントにリダイレクトします。一般公開されるまでアクセスが制限される可能性がありますのでご注意ください。"
          },
          "visit_anyway": {
            "en": "Visit Anyway",
            "jp": "一応見てみる"
          },
          "theme": {
            "en": "Theme",
            "jp": "テーマ"
          },
          "enter_username": {
            "en": "Enter your username",
            "jp": "ユーザー名を入力してください"
          },
          "enter_password": {
            "en": "Enter your password",
            "jp": "パスワードを入力してください"
          },
          "confirm_password_placeholder": {
            "en": "Confirm password",
            "jp": "パスワードを確認してください"
          },
          "confirm_password": {
            "en": "Confirm Password",
            "jp": "パスワードの確認"
          },
  };
  
  export const t = (key: string, language: 'en' | 'jp' = 'jp') => {
    return translations[key]?.[language] || key;
};
  