"""Instructions given to the model for each document type."""

CORD_PROMPT = """Extract the content of this receipt as a single JSON object.
Use only these keys and omit any key that does not appear on the receipt:
- "menu": list of purchased items, each with "nm" (item name), "num" (item code),
  "unitprice", "cnt" (quantity), "discountprice", "price" (line total), "itemsubtotal",
  "vatyn", "etc", and "sub" (list of sub-items with "nm", "unitprice", "cnt", "price", "etc").
- "void_menu": list of cancelled items with "nm" and "price".
- "sub_total": object with "subtotal_price", "discount_price", "service_price",
  "othersvc_price", "tax_price", "etc".
- "total": object with "total_price", "total_etc", "cashprice", "changeprice",
  "creditcardprice", "emoneyprice", "menutype_cnt", "menuqty_cnt".
Copy every value as a string exactly as printed. Answer with the JSON only."""

INVOICE_PROMPT = """Extract the content of this French invoice as a single JSON object
with these keys:
- "invoice_number": string
- "date": issue date as "YYYY-MM-DD"
- "seller": object with "name", "address", "siret" (14 digits), "vat_number"
- "client": object with "name", "address"
- "items": list of lines, each with "description", "quantity" (number),
  "unit_price" (number, excluding VAT), "amount" (number, excluding VAT)
- "total_ht": number, total excluding VAT
- "vat_rate": number, VAT rate in percent
- "vat_amount": number
- "total_ttc": number, total including VAT
Write addresses on a single line separated by ", ". Answer with the JSON only."""

PROMPTS = {"cord": CORD_PROMPT, "invoice": INVOICE_PROMPT}


def build_messages(kind, answer=None):
    """Chat messages for one document; the answer is added for training."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": PROMPTS[kind]}],
        }
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages
