import os
import sqlite3
from thefuzz import fuzz

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'db', 'wilayah_indonesia.sqlite'
)


class WilayahLookup:
    """Fuzzy lookup against the Kemendagri wilayah reference database.

    Each level is matched scoped to its parent (provinsi -> kabupaten ->
    kecamatan -> kelurahan/desa) so OCR typos are corrected against a small,
    contextually-relevant candidate set instead of the full national list.
    """

    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.available = os.path.exists(db_path)
        self.conn = None
        if self.available:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row

    def _query(self, sql, params=()):
        return self.conn.execute(sql, params).fetchall()

    def _best_match(self, text, rows, threshold):
        if not rows:
            return None

        candidates = [(row['name'] or '', row) for row in rows if row['name']]
        if not candidates:
            return None

        names = [c[0] for c in candidates]

        # token_set_ratio handles word-order/subset differences well (e.g.
        # a missing "KABUPATEN"/"KEPULAUAN" qualifier) but scores merged
        # words poorly ("KALIANYAR" vs "KALI ANYAR"), a common OCR artifact
        # where two boxes get read as one. Plain ratio covers that case, so
        # each candidate takes whichever scorer rates it higher.
        scored = [
            (name, max(fuzz.token_set_ratio(text, name), fuzz.ratio(text, name)))
            for name in names
        ]
        if not scored:
            return None

        top_score = max(score for _, score in scored)
        if top_score < threshold:
            return None

        # token_set_ratio treats "RIAU" as a perfect match against
        # "KEPULAUAN RIAU" (subset tokens), so several names can tie at the
        # top score. Break ties with a whole-string ratio, which favors the
        # candidate closest in length/content to the actual OCR text.
        tied_names = {name for name, score in scored if score == top_score}
        best_name = max(tied_names, key=lambda name: fuzz.ratio(text, name))

        row = next(row for name, row in candidates if name == best_name)
        return {'row': row, 'name': best_name, 'score': top_score}

    def _match_scoped(self, text, queries):
        """Try each (sql, params, threshold) in order, returning the first
        match found. Threshold is set per-tier: a query scoped to the
        immediate parent (small, contextually-correct candidate set) can
        safely use a lower bar than a fallback scoped to a grandparent or
        run nationwide, where a wrong guess is more costly."""
        if not self.available or not text:
            return None

        text = text.upper()
        for sql, params, threshold in queries:
            if any(p is None for p in params):
                continue
            rows = self._query(sql, params)
            match = self._best_match(text, rows, threshold)
            if match:
                return match
        return None

    def match_provinsi(self, text, threshold=80):
        return self._match_scoped(text, [
            ("SELECT id, nama_dagri AS name FROM m_provinsi", (), threshold),
        ])

    def match_kabupaten(self, text, id_prov=None, tipe=None):
        """tipe (1=kabupaten, 2=kota) disambiguates regions that exist as
        both, e.g. Kota Bekasi vs Kabupaten Bekasi - pass it whenever the
        OCR text still had its KOTA/KABUPATEN prefix."""
        tipe_and = "AND tipe = ?" if tipe is not None else ""
        tipe_where = "WHERE tipe = ?" if tipe is not None else ""
        tipe_params = (tipe,) if tipe is not None else ()

        return self._match_scoped(text, [
            (
                "SELECT id, id_prov, nama_dagri AS name FROM m_kabupaten "
                f"WHERE id_prov = ? {tipe_and}",
                (id_prov,) + tipe_params, 75
            ),
            (
                f"SELECT id, id_prov, nama_dagri AS name FROM m_kabupaten {tipe_where}",
                tipe_params, 85
            ),
        ])

    def match_kecamatan(self, text, id_kab=None, id_prov=None):
        return self._match_scoped(text, [
            (
                "SELECT id, id_kab, id_prov, nama_dagri AS name "
                "FROM m_kecamatan WHERE id_kab = ?", (id_kab,), 60
            ),
            (
                "SELECT id, id_kab, id_prov, nama_dagri AS name "
                "FROM m_kecamatan WHERE id_prov = ?", (id_prov,), 80
            ),
        ])

    def match_pekerjaan(self, text, threshold=70):
        return self._match_scoped(text, [
            (
                "SELECT id, kode, kategori, nama_pekerjaan AS name "
                "FROM m_pekerjaan", (), threshold
            ),
        ])

    def match_kelurahan(self, text, id_kec=None, id_kab=None, id_prov=None):
        select = (
            "SELECT id, id_kec, id_kab, id_prov, "
            "COALESCE(nama_dagri, nama_bps) AS name FROM m_kelurahan"
        )
        return self._match_scoped(text, [
            (f"{select} WHERE id_kec = ?", (id_kec,), 55),
            (f"{select} WHERE id_kab = ?", (id_kab,), 70),
            (f"{select} WHERE id_prov = ?", (id_prov,), 85),
        ])
