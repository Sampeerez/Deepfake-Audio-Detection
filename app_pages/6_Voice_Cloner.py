# -*- coding: utf-8 -*-
"""
app_pages/6_Voice_Cloner.py, The offensive counterpart to Detection Analysis.

The whole thesis is that a voice can be cloned from just a few seconds of real
audio. This page lets a visitor prove it on themselves: record or upload a short
sample, type any sentence, and hear their own voice say it, in any of the
supported languages. The synthesis is zero-shot voice cloning, delegated to a
Hugging Face Space (Streamlit Community Cloud is CPU-only and cannot host a TTS
model), reached over gradio_client.

The engine is OmniVoice (k2-fsa/OmniVoice), a 2026 multilingual zero-shot cloner.
Every knob its own Gradio demo exposes is surfaced here too, so the clone can be
tuned live: reference transcript, a free-form delivery instruction, and the full
generation settings (inference steps, guidance scale, speed, fixed duration,
denoise, and reference pre/post-processing).

The reference sample is kept in session state, so once a voice is captured you
can retype the text and regenerate as many times as you like. Each result can be
sent, in one click, to Signal Explorer or to Detection Analysis (which analyses
it automatically on arrival).

set_page_config and PAGE_CSS are applied in app.py.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import streamlit as st  # noqa: E402

from src.features import AudioLoadError  # noqa: E402
from src.ui_helpers import get_extractor, mini_note  # noqa: E402

_OMNIVOICE_SPACE = "k2-fsa/OmniVoice"

_LANGUAGES = {
    "Arabic": "ar", "Chinese": "zh", "Danish": "da", "Dutch": "nl",
    "English": "en", "Finnish": "fi", "French": "fr", "German": "de",
    "Greek": "el", "Hebrew": "he", "Hindi": "hi", "Italian": "it",
    "Japanese": "ja", "Korean": "ko", "Malay": "ms", "Norwegian": "no",
    "Polish": "pl", "Portuguese": "pt", "Russian": "ru", "Spanish": "es",
    "Swahili": "sw", "Swedish": "sv", "Turkish": "tr",
}
_CODE_TO_NAME = {v: k for k, v in _LANGUAGES.items()}
_AUTO = "Auto-detect"

_ALL_LANGUAGES = [
    'Abadi', 'Abkhazian', 'Abron', 'Abua', 'Adamawa Fulfulde', 'Adyghe',
    'Afade', 'Afrikaans', 'Agwagwune', 'Aja (Benin)', 'Akebu', 'Alago',
    'Albanian', 'Algerian Arabic', 'Algerian Saharan Arabic',
    'Ambo-Pasco Quechua', 'Ambonese Malay', 'Amdo Tibetan', 'Amharic',
    'Anaang', 'Angika', 'Antankarana Malagasy', 'Aragonese',
    'Arbëreshë Albanian', 'Arequipa-La Unión Quechua', 'Armenian', 'Ashe',
    'Ashéninka Perené', 'Askopan', 'Assamese', 'Asturian', 'Atayal', 'Awak',
    'Ayacucho Quechua', 'Azerbaijani', 'Baatonum', 'Bacama', 'Bade', 'Bafia',
    'Bafut', 'Bagirmi Fulfulde', 'Bago-Kusuntu', 'Baharna Arabic', 'Bakoko',
    'Balanta-Ganja', 'Balti', 'Bamenyam', 'Bamun', 'Bangwinji', 'Banjar',
    'Bankon', 'Baoulé', 'Bara Malagasy', 'Barok', 'Basa (Cameroon)',
    'Basa (Nigeria)', 'Bashkir', 'Basque', 'Batak Mandailing', 'Batanga',
    'Bateri', 'Bats', 'Bayot', 'Bebele', 'Belarusian', 'Bengali', 'Betawi',
    'Bhili', 'Bhojpuri', 'Bilur', 'Bima', 'Bodo', 'Boghom', 'Bokyi', 'Bomu',
    'Bondei', 'Borgu Fulfulde', 'Bosnian', 'Brahui', 'Braj', 'Breton',
    'Buduma', 'Buginese', 'Bukharic', 'Bulgarian', 'Bulu (Cameroon)',
    'Bundeli', 'Bunun', 'Bura-Pabir', 'Burak', 'Burmese', 'Burushaski',
    'Cacaloxtepec Mixtec', 'Cajatambo North Lima Quechua', 'Cakfem-Mushere',
    'Cameroon Pidgin', 'Campidanese Sardinian', 'Cantonese', 'Catalan',
    'Cebuano', 'Cen', 'Central Kurdish', 'Central Nahuatl', 'Central Pame',
    'Central Pashto', 'Central Puebla Nahuatl', 'Central Tarahumara',
    'Central Yupik', 'Central-Eastern Niger Fulfulde', 'Chadian Arabic',
    'Chichewa', 'Chichicapan Zapotec', 'Chiga', 'Chimalapa Zoque',
    'Chimborazo Highland Quichua', 'Chinese', 'Chiquián Ancash Quechua',
    'Chitwania Tharu', 'Chokwe', 'Chuvash', 'Cibak', 'Coastal Konjo',
    'Copainalá Zoque', 'Cornish', 'Corongo Ancash Quechua', 'Croatian',
    'Cross River Mbembe', 'Cuyamecalco Mixtec', 'Czech', 'Dadiya', 'Dagbani',
    'Dameli', 'Danish', 'Dargwa', 'Dazaga', 'Deccan', 'Degema',
    'Dera (Nigeria)', 'Dghwede', 'Dhatki', 'Dhivehi', 'Dhofari Arabic',
    'Dijim-Bwilim', 'Dogri', 'Domaaki', 'Dotyali', 'Duala', 'Dutch', 'DũYa',
    'Dyula', 'Eastern Balochi', 'Eastern Bolivian Guaraní',
    'Eastern Egyptian Bedawi Arabic', 'Eastern Krahn', 'Eastern Mari',
    'Eastern Yiddish', 'Ebrié', 'Eggon', 'Egyptian Arabic', 'Ejagham',
    'Eleme', 'Eloyi', 'Embu', 'English', 'Erzya', 'Esan', 'Esperanto',
    'Estonian', 'Eton (Cameroon)', 'Ewondo', 'Extremaduran',
    'Fang (Equatorial Guinea)', 'Fanti', 'Farefare', "Fe'fe'", 'Filipino',
    'Filomena Mata-Coahuitlán Totonac', 'Finnish', 'Fipa', 'French', 'Fulah',
    'Galician', 'Gambian Wolof', 'Ganda', 'Garhwali', 'Gawar-Bati', 'Gawri',
    'Gbagyi', 'Gbari', 'Geji', 'Gen', 'Georgian', 'German', 'Geser-Gorom',
    'Gheg Albanian', "Ghomálá'", 'Gidar', 'Glavda', 'Goan Konkani', 'Goaria',
    'Goemai', 'Gola', 'Greek', 'Guarani', 'Guduf-Gava', 'Guerrero Amuzgo',
    'Gujarati', 'Gujari', 'Gulf Arabic', 'Gurgula', 'Gusii', 'Gusilay',
    'Gweno', 'Güilá Zapotec', 'Hadothi', 'Hahon', 'Haitian', 'Hakha Chin',
    'Hakö', 'Halia', 'Hausa', 'Hawaiian', 'Hazaragi', 'Hebrew', 'Hemba',
    'Herero', 'Highland Konjo', 'Hijazi Arabic', 'Hindi', 'Huarijio',
    'Huautla Mazatec', 'Huaxcaleca Nahuatl', 'Huba', 'Huitepec Mixtec',
    'Hula', 'Hungarian', 'Hunjara-Kaina Ke', 'Hwana', 'Ibibio', 'Icelandic',
    'Idakho-Isukha-Tiriki', 'Idoma', 'Igbo', 'Igo', 'Ikposo', 'Ikwere',
    'Imbabura Highland Quichua', 'Indonesian', 'Indus Kohistani',
    'Interlingua (International Auxiliary Language Association)', 'Inupiaq',
    'Irish', 'Iron Ossetic', 'Isekiri', 'Isoko', 'Italian', 'Ito', 'Itzá',
    'Ixtayutla Mixtec', 'Izon', 'Jambi Malay', 'Japanese', 'Jaqaru',
    'Jauja Wanca Quechua', 'Jaunsari', 'Javanese', 'Jiba', 'Jju',
    'Judeo-Moroccan Arabic', 'Juxtlahuaca Mixtec', 'Kabardian', 'Kabras',
    'Kabuverdianu', 'Kabyle', 'Kachi Koli', 'Kairak', 'Kalabari', 'Kalasha',
    'Kalenjin', 'Kalkoti', 'Kamba', 'Kamo', 'Kanauji', 'Kanembu', 'Kannada',
    'Karekare', 'Kashmiri', 'Kathoriya Tharu', 'Kati', 'Kazakh', 'Keiyo',
    'Khams Tibetan', 'Khana', 'Khetrani', 'Khmer', 'Khowar', 'Kinga',
    'Kinnauri', 'Kinyarwanda', 'Kirghiz', 'Kirya-Konzəl', 'Kochila Tharu',
    'Kohistani Shina', 'Kohumono', 'Kok Borok', 'Kol (Papua New Guinea)',
    'Kom (Cameroon)', 'Koma', 'Konkani', 'Konzo', 'Korean', 'Korwa',
    'Kota (India)', 'Koti', 'Kuanua', 'Kuanyama', 'Kui (India)',
    'Kulung (Nigeria)', 'Kuot', 'Kushi', 'Kwambi', 'Kwasio', 'Lala-Roba',
    'Lamang', 'Lao', 'Larike-Wakasihu', 'Lasi', 'Latgalian', 'Latvian',
    'Levantine Arabic', 'Liana-Seti', 'Liberia Kpelle', 'Liberian English',
    'Libyan Arabic', 'Ligurian', 'Lijili', 'Lingala', 'Lithuanian', 'Loarki',
    'Logooli', 'Logudorese Sardinian', 'Loja Highland Quichua', 'Loloda',
    'Longuda', 'Loxicha Zapotec', 'Luba-Lulua', 'Luo', 'Lushai',
    'Luxembourgish', 'Maasina Fulfulde', 'Maba (Chad)', 'Macedo-Romanian',
    'Macedonian', 'Mada (Cameroon)', 'Mafa', 'Maithili', 'Malay', 'Malayalam',
    'Mali', "Malinaltepec Me'phaa", 'Maltese', 'Mandara', 'Mandjak',
    'Manggarai', 'Manipuri', 'Mansoanka', 'Manx', 'Maori', 'Marathi',
    'Marghi Central', 'Marghi South', 'Maria (India)', 'Marwari (Pakistan)',
    'Masana', 'Masikoro Malagasy', 'Matsés', 'Mazaltepec Zapotec',
    'Mazatlán Mazatec', 'Mazatlán Mixe', 'Mbe', 'Mbo (Cameroon)', 'Mbum',
    'Medumba', 'Mekeo', 'Meru', 'Mesopotamian Arabic', 'Mewari',
    'Min Nan Chinese', 'Mingrelian', 'Mitlatongo Mixtec', 'Miya', 'Mokpwe',
    'Moksha', 'Mom Jango', 'Mongolian', 'Moroccan Arabic', 'Motu', 'Mpiemo',
    'Mpumpong', 'Mundang', 'Mungaka', 'Musey', 'Musgu', 'Musi', 'Naba',
    'Najdi Arabic', 'Nalik', 'Nawdm', 'Ndonga', 'Neapolitan', 'Nepali',
    'Ngamo', 'Ngas', 'Ngiemboon', 'Ngizim', 'Ngomba', 'Ngombale',
    'Nigerian Fulfulde', 'Nigerian Pidgin', 'Nimadi', 'Nobiin',
    'North Mesopotamian Arabic', 'North Moluccan Malay',
    'Northern Betsimisaraka Malagasy', 'Northern Hindko', 'Northern Kurdish',
    'Northern Pame', 'Northern Pashto', 'Northern Uzbek', 'Northwest Gbaya',
    'Norwegian', 'Norwegian Bokmål', 'Norwegian Nynorsk', 'Notsi', 'Nyankpa',
    'Nyungwe', 'Nzanyi', 'Nüpode Huitoto', 'Occitan', 'Od', 'Odia', 'Odual',
    'Omani Arabic', 'Orizaba Nahuatl', 'Orma', 'Ormuri', 'Oromo',
    'Pahari-Potwari', 'Paiwan', 'Panjabi', 'Papuan Malay', 'Parkari Koli',
    'Pedi', 'Pero', 'Persian', 'Petats', 'Phalura', 'Piemontese',
    'Piya-Kwonci', 'Plateau Malagasy', 'Polish', 'Poqomam', 'Portuguese',
    'Pulaar', 'Pular', 'Puno Quechua', 'Pushto', 'Pökoot', 'Qaqet',
    'Quiotepec Chinantec', 'Rana Tharu', 'Rangi', 'Rapoisi', 'Ratahan',
    'Rayón Zoque', 'Romanian', 'Romansh', 'Rombo', 'Rotokas', 'Rukai',
    'Russian', 'Sacapulteco', 'Saidi Arabic', 'Sakalava Malagasy', 'Sakizaya',
    'Saleman', 'Samba Daka', 'Samba Leko', 'San Felipe Otlaltepec Popoloca',
    'San Francisco Del Mar Huave', 'San Juan Atzingo Popoloca',
    'San Martín Itunyoso Triqui', 'San Miguel El Grande Mixtec', 'Sansi',
    'Sanskrit', 'Santa Ana de Tusi Pasco Quechua',
    'Santa Catarina Albarradas Zapotec', 'Santali',
    'Santiago del Estero Quichua', 'Saposa', 'Saraiki', 'Sardinian', 'Saya',
    'Sediq', 'Serbian', 'Seri', 'Shina', 'Shona', 'Siar-Lak', 'Sibe',
    'Sicilian', 'Sihuas Ancash Quechua', 'Sikkimese', 'Sinaugoro', 'Sindhi',
    'Sindhi Bhil', 'Sinhala', 'Sinicahua Mixtec', 'Sipacapense', 'Siwai',
    'Slovak', 'Slovenian', 'Solos', 'Somali', 'Soninke', 'South Giziga',
    'South Ucayali Ashéninka', 'Southeastern Nochixtlán Mixtec',
    'Southern Betsimisaraka Malagasy', 'Southern Pashto',
    'Southern Pastaza Quechua', 'Soyaltepec Mazatec', 'Spanish',
    'Standard Arabic', 'Standard Moroccan Tamazight', 'Sudanese Arabic',
    'Sulka', 'Svan', 'Swahili', 'Swedish', "Tae'", 'Tahaggart Tamahaq',
    'Taita', 'Tajik', 'Tamil', 'Tandroy-Mahafaly Malagasy', 'Tangale',
    'Tanosy Malagasy', 'Tarok', 'Tatar', 'Tedaga', 'Telugu', 'Tem', 'Teop',
    'Tepeuxila Cuicatec', 'Tepinapa Chinantec', 'Tera', 'Terei', 'Termanu',
    'Tesaka Malagasy', 'Tetelcingo Nahuatl', 'Teutila Cuicatec', 'Thai',
    'Tibetan', 'Tidaá Mixtec', 'Tidore', 'Tigak', 'Tigre', 'Tigrinya',
    'Tilquiapan Zapotec', 'Tinputz', "Tlacoapa Me'phaa",
    'Tlacoatzintepec Chinantec', 'Tlingit', 'Toki Pona', 'Tomoip', 'Tondano',
    'Tonsea', 'Tooro', 'Torau', 'Torwali', 'Tsimihety Malagasy', 'Tsotso',
    'Tswana', 'Tugen', 'Tuki', 'Tula', 'Tulu', 'Tunen', 'Tungag',
    'Tunisian Arabic', 'Tupuri', 'Turkana', 'Turkish', 'Turkmen',
    'Tututepec Mixtec', 'Twi', 'Ubaghara', 'Uighur', 'Ukrainian', 'Umbundu',
    'Upper Sorbian', 'Urdu', 'Ushojo', 'Uzbek', 'Vai', 'Vietnamese', 'Votic',
    'Võro', 'Waci Gbe', 'Wadiyara Koli', 'Waja', 'Wakhi', 'Wanga', 'Wapan',
    'Warji', 'Welsh', 'Wemale', 'Western Frisian',
    'Western Highland Purepecha', 'Western Juxtlahuaca Mixtec',
    'Western Maninkakan', 'Western Mari', 'Western Niger Fulfulde',
    'Western Panjabi', 'Wolof', 'Wuzlam', 'Xanaguía Zapotec', 'Xhosa', 'Yace',
    'Yakut', 'Yalahatan', 'Yanahuanca Pasco Quechua', 'Yangben', 'Yaqui',
    'Yauyos Quechua', 'Yekhee', 'Yiddish', 'Yidgha', 'Yoruba',
    'Yutanduchi Mixtec', 'Zacatlán-Ahuacatlán-Tepetzintla Nahuatl', 'Zarma',
    'Zaza', 'Zulu', 'Ömie',
]

_MAX_REF_SECONDS = 30
_MIN_REF_SECONDS = 3.0
_MAX_TEXT_CHARS = 600

_ENDPOINT_CACHE = {}


def _secret(key, default=None):
    """Read an optional Streamlit secret without exploding when no secrets file
    exists (the app ships without one)."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def _detect_language(text: str) -> str:
    """Best-effort language id for the typed text, restricted to the display
    subset, used only to label the result nicely when the language is Auto.
    Uses langdetect when available, with a light Spanish/English fallback."""
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        code = detect(text)
        code = {"zh-cn": "zh", "zh-tw": "zh"}.get(code, code)
        if code in _CODE_TO_NAME:
            return code
    except Exception:
        pass
    low = f" {text.lower()} "
    _es_marks = "ñ¿¡áéíóú"
    _es_words = (" el ", " la ", " que ", " de ", " es ", " hola ", " gracias ",
                 " mi ", " tu ", " los ", " una ", " por ", " con ")
    if any(c in low for c in _es_marks) or any(w in low for w in _es_words):
        return "es"
    return "en"


@st.cache_resource(show_spinner=False)
def _get_client(space_id: str, hf_token):
    """Connect to the cloning Space once and reuse the connection across reruns.
    A failed connection raises here (and is not cached), so a later retry can
    succeed once the Space wakes up. The token kwarg was renamed across
    gradio_client releases (hf_token -> token), so support both."""
    from gradio_client import Client
    try:
        return Client(space_id, token=hf_token or None)
    except TypeError:
        return Client(space_id, hf_token=hf_token or None)


def _clone_endpoint(client, space_id):
    """The OmniVoice demo exposes its clone handler under the click function's
    name, /_clone_fn on current builds. Stay resilient to a rename by asking the
    Space for its endpoints and picking the clone one, falling back to the known
    name if discovery is unavailable."""
    if space_id in _ENDPOINT_CACHE:
        return _ENDPOINT_CACHE[space_id]
    api = "/_clone_fn"
    try:
        info = client.view_api(print_info=False, return_format="dict") or {}
        eps = list(info.get("named_endpoints", {}).keys())
        if "/_clone_fn" not in eps:
            api = next((e for e in eps if "clone" in e.lower()),
                       eps[0] if eps else api)
    except Exception:
        pass
    _ENDPOINT_CACHE[space_id] = api
    return api


def _parse_out(out):
    """OmniVoice's clone endpoint returns (audio, status): the audio comes back
    as a local filepath gradio_client already downloaded, the status is a short
    string ("Done." or an "Error: ..."/"Please ..." message). Split them."""
    if isinstance(out, (tuple, list)):
        audio = out[0] if out else None
        status = out[1] if len(out) > 1 else ""
        return (audio if isinstance(audio, str) else None), (status or "")
    return (out if isinstance(out, str) else None), ""


def _synthesize(hf_token, ref_wav_path, text, lang_name, ref_text, instruct,
                steps, cfg, denoise, speed, duration, preprocess, postprocess):
    """Run OmniVoice's zero-shot clone endpoint, return a local wav path.

    The argument order mirrors the Space's own _clone_fn(text, lang, ref_audio,
    ref_text, instruct, steps, cfg, denoise, speed, duration, preprocess,
    postprocess). duration 0 means "let speed decide" (the Space treats <= 0 as
    unset). The Space can be redirected via the OMNIVOICE_SPACE secret."""
    from gradio_client import handle_file

    space = _secret("OMNIVOICE_SPACE") or _OMNIVOICE_SPACE
    client = _get_client(space, hf_token)
    api = _clone_endpoint(client, space)
    ref = handle_file(ref_wav_path)

    out = client.predict(
        text, lang_name, ref, ref_text or "", instruct or "",
        int(steps), float(cfg), bool(denoise), float(speed),
        float(duration or 0), bool(preprocess), bool(postprocess),
        api_name=api)

    audio_path, status = _parse_out(out)
    if not audio_path:
        raise RuntimeError(status or "The engine returned no audio.")
    return audio_path


def _write_reference(blob: bytes, name) -> str:
    """Prepare the best-possible reference for the cloner.

    A clean reference is the single biggest lever on clone fidelity, so instead
    of the detector pipeline's 16 kHz (which throws away the treble the cloner
    needs) we:
      1. decode at the model's native 24 kHz, keeping the full timbre,
      2. high-pass out the sub-bass rumble/DC that real mic recordings carry,
      3. trim leading/trailing silence so the voice print is not diluted,
      4. peak-normalise so the reference is at a consistent, healthy level,
      5. cap the length so the Space is never overloaded.
    """
    import io as _io

    import librosa
    try:
        y, _ = librosa.load(_io.BytesIO(blob), sr=24000, mono=True)
        sr_out = 24000
    except Exception:
        y = get_extractor().load_audio_bytes(blob, name)
        sr_out = get_extractor().sample_rate

    try:
        from scipy.signal import butter, sosfilt
        sos = butter(4, 70, btype="highpass", fs=sr_out, output="sos")
        y = sosfilt(sos, y).astype(np.float32)
    except Exception:
        pass

    try:
        y_trim, _ = librosa.effects.trim(y, top_db=35)
        if y_trim.size >= sr_out * 0.5:
            y = y_trim
    except Exception:
        pass

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = (0.97 / peak) * y

    y = y[: int(_MAX_REF_SECONDS * sr_out)]
    fh = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    fh.close()
    sf.write(fh.name, y.astype(np.float32), sr_out)
    return fh.name


def _clear_reference():
    """Forget the captured voice AND reset the input widgets.

    Both st.tabs bodies run every rerun, so clearing only the mirrored payload
    keys is not enough: whichever widget (recorder or uploader) still holds a
    file re-writes vc_ref_bytes on the very next run, and Clear appears to do
    nothing. Popping the widget keys here, in an on_click callback (before the
    widgets re-instantiate), makes them come back empty. Same fix pattern as
    Signal Explorer's _clear_upload."""
    for _k in ("vc_ref_bytes", "vc_ref_name", "vc_out_bytes", "vc_out_name",
               "vc_out_lang", "vc_recorder", "vc_ref_upload"):
        st.session_state.pop(_k, None)


st.title("Voice Cloner")
st.markdown(
    "A few seconds of real audio are **enough to clone a voice**. Record or "
    "upload a short sample of yourself, type any sentence, and hear it spoken "
    "back in your own voice. Then send the result to **Signal Explorer** or "
    "**Detection Analysis** and see how the detectors react. That is the whole "
    "point: detection exists because cloning is this easy."
)
st.caption(
    "Cloning runs on OmniVoice. Use this only on your own voice, or a voice you "
    "have explicit consent to use. It is an educational demonstration of why "
    "synthetic speech detection matters, not a tool to impersonate anyone."
)


extractor = get_extractor()
_sr = extractor.sample_rate

with st.container(border=True):
    st.markdown('<div class="section-label" style="margin-bottom:0rem;">'
                'Your voice</div>', unsafe_allow_html=True)

    _tab_rec, _tab_up = st.tabs(["Record", "Upload"])
    with _tab_rec:
        _rec = st.audio_input("Record a few seconds of your voice",
                              key="vc_recorder", label_visibility="collapsed")
        if _rec is not None:
            _blob = _rec.getvalue()
            if _blob != st.session_state.get("vc_ref_bytes"):
                st.session_state["vc_ref_bytes"] = _blob
                st.session_state["vc_ref_name"] = "recording.wav"
    with _tab_up:
        _up = st.file_uploader(
            "Upload", type=["flac", "wav", "mp3", "ogg", "m4a"],
            key="vc_ref_upload", label_visibility="collapsed")
        if _up is not None:
            _blob = _up.getvalue()
            if _blob != st.session_state.get("vc_ref_bytes"):
                st.session_state["vc_ref_bytes"] = _blob
                st.session_state["vc_ref_name"] = _up.name

    _ref_bytes = st.session_state.get("vc_ref_bytes")
    _ref_name = st.session_state.get("vc_ref_name")
    _ref_signal = None

    if _ref_bytes is None:
        st.caption("For the best clone: record about 10 to 15 seconds in a quiet "
                   "room, speaking naturally and with a bit of expression, no "
                   "background music or noise. Reference quality is what makes or "
                   "breaks the result.")
    else:
        try:
            _ref_signal = extractor.load_audio_bytes(_ref_bytes, _ref_name)
        except AudioLoadError as exc:
            st.error(f"Could not read that audio: {exc}")

        if _ref_signal is not None:
            _dur = len(_ref_signal) / _sr
            _c_play, _c_clear = st.columns([6, 1], vertical_alignment="center")
            with _c_play:
                st.audio(_ref_signal, sample_rate=_sr)
            with _c_clear:
                st.button("Clear", icon=":material/close:", width="stretch",
                          key="vc_clear", on_click=_clear_reference)
            st.caption(f"Voice captured from **{_ref_name}** ({_dur:.1f} s)")
            if _dur < _MIN_REF_SECONDS:
                mini_note(
                    f"That sample is only {_dur:.1f} s. It may still work, but "
                    f"{_MIN_REF_SECONDS:.0f} s or more clones far more "
                    "faithfully.", warn=True)


if _ref_signal is not None:
    with st.container(border=True):
        st.markdown('<div class="section-label" style="margin-bottom:0.9rem;">'
                    'Make it say anything</div>', unsafe_allow_html=True)

        _text = st.text_area(
            "Text to speak", key="vc_text", height=110,
            max_chars=_MAX_TEXT_CHARS,
            placeholder="Type a sentence here and the cloned voice will say it. "
                        "Retype it and clone again as many times as you like...",
            label_visibility="collapsed")

        _c_lang, _c_ref = st.columns(2)
        with _c_lang:
            _lang_choice = st.selectbox(
                "Language", [_AUTO] + _ALL_LANGUAGES, key="vc_lang",
                help="Auto-detect reads the language from your text. You can "
                     "also type to search any of OmniVoice's 600+ languages.")
        with _c_ref:
            _ref_text = st.text_input(
                "Reference transcript (optional)", key="vc_ref_text",
                placeholder="What your recording says, word for word...",
                help="OmniVoice clones more faithfully when it knows what the "
                     "reference says. Leave empty to let it auto-transcribe.")

        _instruct = st.text_input(
            "Instruct (optional)", key="vc_instruct",
            placeholder="Optional delivery hint, e.g. 'speak slowly and "
                        "calmly'...")

    with st.container(border=True):
        st.markdown('<div class="section-label" style="margin-bottom:0.2rem;">'
                    'Fine-tuning</div>', unsafe_allow_html=True)
        st.caption("These already have good defaults, you can clone without "
                   "touching any of them. Adjust only if you want to experiment.")

        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            _steps = st.slider("Inference steps", 4, 64, 32, 1, key="vc_steps")
            st.caption("Higher is crisper but slower. Default 32.")
        with _c2:
            _cfg = st.slider("Guidance scale", 0.0, 4.0, 2.0, 0.1, key="vc_cfg")
            st.caption("Higher sticks closer to the text. Default 2.0.")
        with _c3:
            _speed = st.slider("Speed", 0.5, 1.5, 1.0, 0.05, key="vc_speed")
            st.caption("Above 1 faster, below 1 slower. Default 1.0.")

        with st.container(key="vc_ft_toggles"):
            _c4, _c5, _c6, _c7 = st.columns([1.6, 1, 1, 1], gap="medium")
            with _c4:
                _duration = st.number_input(
                    "Fixed duration (s)", min_value=0.0, value=0.0, step=0.5,
                    key="vc_duration")
                st.caption("Force the clip length. 0 lets Speed decide.")
            _pad = "<div class='vc-toggle-pad'></div>"
            with _c5:
                st.markdown(_pad, unsafe_allow_html=True)
                _denoise = st.checkbox("Denoise", value=True, key="vc_denoise")
                st.caption("Clean noise from the output.")
            with _c6:
                st.markdown(_pad, unsafe_allow_html=True)
                _preprocess = st.checkbox("Preprocess reference", value=True,
                                          key="vc_pre")
                st.caption("Trim silence from your recording first.")
            with _c7:
                st.markdown(_pad, unsafe_allow_html=True)
                _postprocess = st.checkbox("Postprocess output", value=True,
                                           key="vc_post")
                st.caption("Remove long silences from the result.")

    _do_clone = st.button(
        "Clone voice", type="primary", icon=":material/graphic_eq:",
        width="stretch", key="vc_clone_btn")

    _hf_token = (_secret("HF_TOKEN") or _secret("HUGGINGFACE_TOKEN")
                 or _secret("HF_API_TOKEN") or os.environ.get("HF_TOKEN"))

    _clean_text = (_text or "").strip()
    if _do_clone and not _clean_text:
        st.warning("Type some text for the voice to say first.")
    elif _do_clone:
        if _lang_choice == _AUTO:
            _lang_name = "Auto"
            _lang_lbl = _CODE_TO_NAME.get(_detect_language(_clean_text),
                                          "Auto-detect")
        else:
            _lang_name = _lang_choice
            _lang_lbl = _lang_choice
        _ref_path = _write_reference(_ref_bytes, _ref_name)
        try:
            with st.spinner("Cloning your voice, this can take a moment..."):
                _out_path = _synthesize(
                    _hf_token, _ref_path, _clean_text, _lang_name, _ref_text,
                    _instruct, _steps, _cfg, _denoise, _speed, _duration,
                    _preprocess, _postprocess)
            with open(_out_path, "rb") as _fh:
                st.session_state["vc_out_bytes"] = _fh.read()
            st.session_state["vc_out_name"] = "cloned_voice.wav"
            st.session_state["vc_out_lang"] = _lang_lbl
        except Exception as exc:  # noqa: BLE001, keep the page alive on failure
            st.session_state.pop("vc_out_bytes", None)
            _msg = str(exc)
            if "quota" in _msg.lower() or "gpu" in _msg.lower():
                if _hf_token:
                    st.warning(
                        "Your Hugging Face account has used up its GPU quota on "
                        "the shared Space for now. It resets after a while, so "
                        "try again later, or point **OMNIVOICE_SPACE** at your "
                        "own duplicated Space (or a local copy) for private "
                        "quota.\n\n"
                        f"Details: {exc}")
                else:
                    st.warning(
                        "The free shared Space ran out of GPU quota for "
                        "anonymous users (it resets after a short wait). To "
                        "clone repeatedly, add a free Hugging Face token as "
                        "**HF_TOKEN** in the app secrets, it raises the quota a "
                        "lot. Get one at huggingface.co/settings/tokens.\n\n"
                        f"Details: {exc}")
            else:
                st.error(
                    "The engine did not answer. Free Hugging Face Spaces sleep "
                    "when idle and can take 30 to 60 seconds to wake up, or may "
                    "be queued, so try again in a moment.\n\n"
                    f"Details: {exc}")
        finally:
            try:
                os.unlink(_ref_path)
            except OSError:
                pass


_out_bytes = st.session_state.get("vc_out_bytes")
if _out_bytes is not None:
    with st.container(border=True):
        _lang_lbl = st.session_state.get("vc_out_lang")
        _suffix = f" &middot; {_lang_lbl}" if _lang_lbl else ""
        st.markdown(f'<div class="section-label" style="margin-bottom:1rem;">Your cloned voice{_suffix}</div>',
            unsafe_allow_html=True)
        st.audio(_out_bytes, format="audio/wav")

        _c_se, _c_da = st.columns(2)
        with _c_se:
            if st.button("View in Signal Explorer", icon=":material/graphic_eq:",
                         width="stretch", key="vc_to_signal"):
                st.session_state["a_upload_name"] = "cloned_voice.wav"
                st.session_state["a_upload_bytes"] = _out_bytes
                st.session_state["a_source"] = "Upload"
                st.switch_page("app_pages/1_Signal_Explorer.py")
        with _c_da:
            if st.button("Test in Detection Analysis", icon=":material/radar:",
                         width="stretch", key="vc_to_detect"):
                st.session_state["da_test_bytes"] = _out_bytes
                st.session_state["da_test_name"] = "cloned_voice.wav"
                st.session_state.pop("da_test_rows", None)
                st.session_state["da_auto_analyze"] = True
                st.switch_page("app_pages/3_Detection_Analysis.py")
