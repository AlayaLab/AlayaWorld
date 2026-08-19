# Generated model client

`client.ts` and `client.react.tsx` are generated from the model's own schema, so
every command, parameter, and message in this app is checked against what the model
actually serves. Do not edit them by hand.

`client.ts` holds the command parameters, the message types, and the track list.
`client.react.tsx` holds the React layer: `useAlayaWorld()` for the typed commands,
one hook per message, and `AlayaWorldMainVideoView` for the video. Both import
`Reactor` from `@reactor-team/js-sdk`, which the app depends on directly.

The `AlayaWorld` in those names is the model's own name, pascal-cased: `model.name`
in `reactor.yaml` is `alaya-world`, and the generator splits it on the hyphen.
Renaming the model renames this whole surface.

The app does not use the generated provider, because that provider fixes the model
name at generation time. It wraps the SDK's own `ReactorProvider` instead, so the
name can come from the environment.

## Regenerating

Start the model, read its schema, and generate from that:

```sh
# from the repository root, with the model running
curl -s localhost:8080/schema -o /tmp/alayaworld-schema.json

npx @reactor-team/codegen \
  --schema /tmp/alayaworld-schema.json \
  --standalone --react \
  --output reactor/demo/lib/alayaworld/client.ts
```

`MODEL_VERSION` in the generated file reflects what the running container reports,
which is a placeholder until the model is released through the platform.
