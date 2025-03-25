m['model']

y['model']


# embedding

#encoder
for key in m['model'].keys():
    if key.startswith("language_model.decoder"):
        print(f'key: {key}')
        fields = key.replace("decoder", "encoder").split(".")
        fields = fields[0], fields[1], ".".join(fields[2:])
        w = y['model']
        for f in fields:
            w = w[f]

        m['model'][key] = w

