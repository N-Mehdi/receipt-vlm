"""Synthetic French invoices: coherent annotations, HTML rendering and light augmentation."""

import datetime
import io
import json
import random
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from faker import Faker
from jinja2 import Environment, FileSystemLoader
from PIL import Image, ImageEnhance, ImageFilter

TEMPLATE_DIR = Path(__file__).parent / "templates"
SEEN_TEMPLATES = ["classic", "modern", "compact", "boxed"]
UNSEEN_TEMPLATES = ["serif", "minimal"]
VAT_RATES = [Decimal("20"), Decimal("10"), Decimal("5.5")]
CENT = Decimal("0.01")
MONTHS = [
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
]
SERVICES = [
    "Prestation de conseil",
    "Licence logiciel annuelle",
    "Maintenance informatique",
    "Formation professionnelle",
    "Développement web",
    "Hébergement serveur",
    "Audit de sécurité",
    "Support technique",
    "Rédaction de contenu",
    "Location de matériel",
    "Installation réseau",
    "Frais de déplacement",
    "Cartouches d'encre",
    "Papier A4 ramette",
    "Chaise de bureau",
    "Écran 27 pouces",
    "Clavier sans fil",
    "Câble HDMI",
    "Nettoyage des locaux",
    "Livraison",
]
DETAILS = ["", "", " standard", " premium", " (forfait)", " - lot de 5", " mensuel", " sur site"]


def luhn_valid(digits):
    """Luhn check used by SIREN and SIRET numbers."""
    total = 0
    for i, char in enumerate(reversed(digits)):
        d = int(char)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def luhn_complete(partial):
    """Append the digit that makes the number Luhn valid."""
    for d in range(10):
        if luhn_valid(partial + str(d)):
            return partial + str(d)
    raise ValueError(partial)


def make_siret(rng):
    """SIREN (9 digits) then NIC (5 digits), both Luhn valid."""
    siren = luhn_complete(str(rng.randint(1, 9)) + "".join(rng.choices("0123456789", k=7)))
    return luhn_complete(siren + "".join(rng.choices("0123456789", k=4)))


def vat_number(siren):
    """French intra community VAT number derived from the SIREN."""
    key = (12 + 3 * (int(siren) % 97)) % 97
    return "FR" + str(key).zfill(2) + siren


def as_number(value):
    """Decimal to int or float for JSON."""
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def make_invoice(fake, rng):
    """Return (annotation, values used for display)."""
    siret = make_siret(rng)
    rate = rng.choice(VAT_RATES)
    items = []
    lines = []
    total_ht = Decimal("0")
    for _ in range(rng.randint(1, 8)):
        quantity = rng.choice([1, 1, 1, 2, 3, 4, 5, 10, 12, 20])
        unit = (Decimal(rng.randint(150, 250000)) / 100).quantize(CENT)
        amount = (unit * quantity).quantize(CENT)
        total_ht += amount
        description = rng.choice(SERVICES) + rng.choice(DETAILS)
        items.append(
            {
                "description": description,
                "quantity": quantity,
                "unit_price": as_number(unit),
                "amount": as_number(amount),
            }
        )
        lines.append(
            {"description": description, "quantity": quantity, "unit": unit, "amount": amount}
        )
    vat = (total_ht * rate / 100).quantize(CENT, rounding=ROUND_HALF_UP)
    total_ttc = total_ht + vat
    date = fake.date_between(datetime.date(2023, 1, 1), datetime.date(2025, 12, 31))
    seller_address = fake.address()
    client_address = fake.address()
    number = rng.choice(["FA-{y}-{n:05d}", "F{y}{n:04d}", "{y}/{n:04d}", "INV-{n:06d}"])
    target = {
        "invoice_number": number.format(y=date.year, n=rng.randint(1, 9999)),
        "date": date.isoformat(),
        "seller": {
            "name": fake.company(),
            "address": seller_address.replace("\n", ", "),
            "siret": siret,
            "vat_number": vat_number(siret[:9]),
        },
        "client": {
            "name": rng.choice([fake.company(), fake.name()]),
            "address": client_address.replace("\n", ", "),
        },
        "items": items,
        "total_ht": as_number(total_ht),
        "vat_rate": as_number(rate),
        "vat_amount": as_number(vat),
        "total_ttc": as_number(total_ttc),
    }
    display = {
        "lines": lines,
        "total_ht": total_ht,
        "rate": rate,
        "vat": vat,
        "total_ttc": total_ttc,
        "date": date,
        "phone": fake.phone_number(),
        "email": fake.company_email(),
        "iban": fake.iban(),
        "due_days": rng.choice([15, 30, 45, 60]),
        "seller_lines": seller_address.split("\n"),
        "client_lines": client_address.split("\n"),
    }
    return target, display


def eur(value):
    """French money format: 1 234,50 €."""
    text = format(Decimal(value).quantize(CENT), ",.2f")
    return text.replace(",", "\u00a0").replace(".", ",") + "\u00a0€"


def fr_number(value):
    return format(Decimal(value).normalize(), "f").replace(".", ",")


def long_date(value):
    return str(value.day) + " " + MONTHS[value.month - 1] + " " + str(value.year)


def make_env():
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True)
    env.filters["eur"] = eur
    env.filters["num"] = fr_number
    env.filters["long_date"] = long_date
    return env


def render_html(env, template, target, display):
    return env.get_template(template + ".html").render(inv=target, d=display)


def augment(image, rng):
    """Slight rotation, contrast, brightness and blur."""
    image = image.convert("RGB")
    image = image.rotate(
        rng.uniform(-1.5, 1.5), expand=True, fillcolor="white", resample=Image.Resampling.BICUBIC
    )
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.1))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.08))
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.8)))
    return image


def split_plan(n_train, n_val, n_test_seen, n_test_unseen):
    """List of (split, template) pairs; unseen templates appear only in test."""
    plan = []
    counts = [
        ("train", n_train, SEEN_TEMPLATES),
        ("validation", n_val, SEEN_TEMPLATES),
        ("test", n_test_seen, SEEN_TEMPLATES),
        ("test", n_test_unseen, UNSEEN_TEMPLATES),
    ]
    for split, n, templates in counts:
        for i in range(n):
            plan.append((split, templates[i % len(templates)]))
    return plan


def generate(out_dir, n_train=1000, n_val=100, n_test_seen=50, n_test_unseen=50, seed=0):
    """Write images and annotations.jsonl into out_dir/<split>/."""
    from playwright.sync_api import sync_playwright

    rng = random.Random(seed)
    fake = Faker("fr_FR")
    fake.seed_instance(seed)
    env = make_env()
    out_dir = Path(out_dir)
    files = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 1200})
        plan = split_plan(n_train, n_val, n_test_seen, n_test_unseen)
        for index, (split, template) in enumerate(plan):
            target, display = make_invoice(fake, rng)
            page.set_content(render_html(env, template, target, display))
            png = page.locator(".page").screenshot()
            image = augment(Image.open(io.BytesIO(png)), rng)
            folder = out_dir / split
            folder.mkdir(parents=True, exist_ok=True)
            name = split + "_" + str(index).zfill(5) + ".jpg"
            image.save(folder / name, quality=rng.randint(70, 95))
            row = {
                "id": name[:-4],
                "file": name,
                "template": template,
                "seen": template in SEEN_TEMPLATES,
                "ground_truth": json.dumps(target, ensure_ascii=False),
            }
            if split not in files:
                files[split] = open(folder / "annotations.jsonl", "w", encoding="utf-8")
            files[split].write(json.dumps(row, ensure_ascii=False) + "\n")
        browser.close()
    for handle in files.values():
        handle.close()
